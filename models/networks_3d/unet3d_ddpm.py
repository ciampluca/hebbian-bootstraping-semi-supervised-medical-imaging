import torch
from torch import nn
from torch.nn import init

from denoising_diffusion_pytorch import *
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import *


class SuperDiffusion(GaussianDiffusion):
    def model_predictions(self, x, t, model_output, clip_x_start = False, rederive_pred_noise = False):
        #model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_losses(self, x_start, y_start, t, loss_fn=None, noise = None, offset_noise_strength = None):
        c_in, c_out = x_start.shape[1], y_start.shape[1]
        #x_start = torch.cat([x_start, y_start], dim=1)
        
        #b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        x = torch.cat([x, y_start], dim=1)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        #y_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                model_out = self.model(x, t)
                x_self_cond = self.model_predictions(x[:, :c_in], t, model_out).pred_x_start
                x_self_cond.detach_()
                #y_self_cond = self.model_predictions(x[:, -c_out:], t, model_out).pred_x_start
                #y_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            #target = self.predict_noise_from_start(x, t, x_start)[:, -c_out:]
            target = self.predict_noise_from_start(x[:, :c_in], t, x_start[:, :c_in])
        elif self.objective == 'pred_x0':
            #target = x_start[:, -c_out:]
            target = x_start[:, :c_in]
        elif self.objective == 'pred_v':
            #target = self.predict_v(x_start, t, noise)[:, -c_out:]
            target = self.predict_v(x_start[:, :c_in], t, noise[:, :c_in])
        else:
            raise ValueError(f'unknown objective {self.objective}')

        #model_out = self.model_predictions(x[:, -c_out:], t, model_out).pred_x_start
        model_pred =  self.model_predictions(x[:, :c_in], t, model_out).pred_x_start
        
        if loss_fn is None: loss = F.mse_loss(model_out, target) #, reduction = 'none')
        else: loss = loss_fn(self.unnormalize(model_pred), torch.argmax(self.unnormalize(x_start[:, :c_in]), dim=1, keepdim=True).to(dtype=torch.long))
        #loss = reduce(loss, 'b ... -> b', 'mean')
        #loss_reconstr = F.mse_loss(model_reconstr, target_reconstr, reduction = 'none')
        #loss_reconstr = reduce(loss_reconstr, 'b ... -> b', 'mean')
        lw = self.loss_weight #if loss_fn is F.mse_loss else self.get_loss_weights('pred_x0')
        loss = loss * extract(lw, t, loss.shape)
        #loss_reconstr = loss_reconstr * extract(self.loss_weight, t, loss_reconstr.shape)
        
        return loss.mean(), model_pred #loss.mean(), loss_reconstr.mean(), model_out, model_reconstr

    def forward(self, img, target, conditioner='target', loss_fn=None, *args, **kwargs):
        if target.ndim == img.ndim - 1: target = target.unsqueeze(1)
        if target.shape[1] == 1: target = torch.cat([target, 1 - target], dim=1)
        target = target.to(dtype=img.dtype)
        b, c, h, w, z, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        #if conditioner =='target': img = self.normalize(img)
        #else: target = self.normalize(target)
        img, target = self.normalize(img), self.normalize(target)
        loss, pred = self.p_losses(img, target, t, loss_fn, *args, **kwargs) if conditioner =='target' else self.p_losses(target, img, t, loss_fn, *args, **kwargs)
        return loss, self.unnormalize(pred)

    def get_loss_weights(self, objective):
        snr = self.alphas_cumprod / (1 - self.alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        #if min_snr_loss_weight:
        #    maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            return maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            return maybe_clipped_snr
        elif objective == 'pred_v':
            return maybe_clipped_snr / (snr + 1)
        else:
            raise ValueError(f'unknown objective {objective}')

    def p_mean_variance(self, x, t, x_start, x_self_cond = None, clip_denoised = True):
        model_out = self.model(x, t, x_self_cond)
        preds = self.model_predictions(x[:, :model_out.shape[1]], t, model_out).pred_x_start
        x_start = torch.cat([preds, x_start[:, preds.shape[1]:]], dim=1)

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x[:, :preds.shape[1]], t = t)
        return model_mean[:, :preds.shape[1]], posterior_variance[:, :preds.shape[1]], posterior_log_variance[:, :preds.shape[1]], x_start[:, :preds.shape[1]]

    def sample_mask_loop(self, x_start, y_start):
        c_in, c_out = x_start.shape[1], y_start.shape[1]
        #x_start = torch.cat([x_start, y_start], dim=1)
        noise = torch.randn_like(x_start)
        b, *_, device = *x_start.shape, x_start.device
        t = torch.full((b,), self.num_timesteps, device = device, dtype = torch.long) - 1
        x = self.q_sample(x_start, t, noise=noise)
        x = torch.cat([x, y_start], dim=1)
        x_cond = None

        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_cond if self.self_condition else None
            b, *_, device = *x.shape, x.device
            batched_times = torch.full((b,), t, device = device, dtype = torch.long)
            model_mean, _, model_log_variance, x_cond = self.p_mean_variance(x = x, t = batched_times, x_start = x_start, x_self_cond = self_cond, clip_denoised = True)
            noise = torch.randn_like(model_mean) if t > 0 else 0. # no noise if t == 0
            pred = model_mean + (0.5 * model_log_variance).exp() * noise
            x = torch.cat([pred, x[:, -c_out:]], dim=1)

        return self.unnormalize(x[:, :c_in])

    def sample_mask(self, img, target, conditioner='target'):
        if target.ndim == img.ndim - 1: target = target.unsqueeze(1)
        if target.shape[1] == 1: target = torch.cat([target, 1 - target], dim=1)
        target = target.to(dtype=img.dtype)

        #if conditioner =='target': img = self.normalize(img)
        #else: target = self.normalize(target)
        img, target = self.normalize(img), self.normalize(target)
        return self.sample_mask_loop(img, target) if conditioner =='target' else self.sample_mask_loop(target, img)


from ..networks_3d.unet3d  import UNet3D, init_weights

class DDPM_Wrapper(Unet):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        #pred_out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        n_classes = 2,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        attn_enabled = True,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False
    ):
        super(Unet, self).__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels + n_classes + (channels if self_condition else 0)
        self.n_classes = n_classes

        init_dim = default(init_dim, dim)
        self.init_conv = UNet3D._block(input_channels, init_dim, name='enc0') #ConvBlockLeakyRelu(input_channels, init_dim, dropout[0]) #nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, init_dim) #time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                nn.MaxPool3d(kernel_size=2, stride=2),
                UNet3D._block(dim_in, dim_out, name='enc{}'.format(ind+1)), #DownBlock(dim_in, dim_out, dropout[ind+1]), #resnet_block(dim_in, dim_in),
                nn.Identity(), #resnet_block(dim_in, dim_in),
                nn.Identity(), #attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads) if attn_enabled else nn.BatchNorm2d(dim_in),  #nn.Identity(),
                #nn.Identity(), #Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = nn.MaxPool3d(kernel_size=2, stride=2) #nn.Identity() #resnet_block(mid_dim, mid_dim)
        self.mid_attn = nn.Identity() #FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1]) if attn_enabled else nn.BatchNorm2d(mid_dim) #nn.Identity()
        self.mid_block2 = UNet3D._block(mid_dim, mid_dim, name='bottleneck{}'.format(len(in_out))) #nn.Identity() #resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                nn.ConvTranspose3d(dim_out, dim_out, kernel_size=2, stride=2),
                UNet3D._block(dim_out + dim_in, dim_in, name='dec{}'.format(len(in_out) - ind)), #UpBlock(dim_out, dim_in, dim_in, dropout[-ind]), #resnet_block(dim_out + dim_in, dim_out),
                nn.Identity(), #resnet_block(dim_out + dim_in, dim_out),
                nn.Identity(), #attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads) if attn_enabled else nn.BatchNorm2d(dim_out), #nn.Identity(),
                #nn.Identity(), #Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        #default_pred_out_dim = n_classes * (1 if not learned_variance else 2)
        #self.pred_out_dim = default(pred_out_dim, default_pred_out_dim)

        self.final_res_block = nn.Identity() #resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv3d(init_dim, self.out_dim, 3, padding=1)
        #self.final_conv_pred = nn.Conv2d(init_dim, self.pred_out_dim, 1)

    def forward(self, x, time, x_self_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        t = self.time_mlp(time)
        x = x + t.reshape(t.shape[0], t.shape[1], 1, 1, 1)
        r = x.clone()

        h = []

        for block1, block2, attn, downsample in self.downs:
            h.append(x)
            x = block1(x) #, t)
            #h.append(x)

            x = block2(x) #, t)
            #x = attn(x) + x
            #h.append(x)

            #x = downsample(x)

        #h.append(x)
        #x = self.mid_block1(x) #, t)
        #x = self.mid_attn(x) + x
        #x = self.mid_block2(x) #, t)

        for block1, block2, attn, upsample in self.ups:
            #x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x) #, h.pop()) #, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x) #, t)
            #x = attn(x) + x

            #x = upsample(x)

        #x = torch.cat((x, r), dim = 1)

        #x = self.final_res_block(x, t)

        return self.final_conv(x) #self.final_conv_pred(x), self.final_conv(x)

class UNet_Wrapper(nn.Module):
    def __init__(self, in_chns, class_num, timestamp_diffusion=1000):
        super().__init__()
        self.net = UNet3D(in_chns, class_num)
        self.channels = in_chns
        self.n_classes = class_num
        self.self_condition = False

    def forward(self, x, time, x_self_cond = None):
        return self.net.forward(x[:, -self.channels:])

class DDPMUNet(nn.Module):

    def __init__(self, in_chns, class_num, timestamp_diffusion=1000):
        super(DDPMUNet, self).__init__()

        self.net = DDPM_Wrapper(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            channels = in_chns,
            n_classes = class_num,
            out_dim=in_chns
        )

        self.net_seg = DDPM_Wrapper(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            channels = class_num,
            n_classes = in_chns,
            out_dim=class_num,
            attn_enabled = False
        )
        #self.net_seg = UNet_Wrapper(in_chns, class_num, timestamp_diffusion)

        self.final_conv = nn.Conv3d(class_num, class_num, 3, padding=1)

    def forward(self, x):
        return self.final_conv(x)

    #def state_dict(self):
    #    return self.net.state_dict()
    
    #def load_state_dict(self, state_dict, strict = ...):
    #    return self.net.load_state_dict(state_dict, strict)
    
    #def reset_clf(self, out_channels):
    #    device = self.net.final_conv_pred.weight.device
    #    self.net.final_conv_pred = nn.Conv2d(self.net.final_conv_pred.weight.shape[1], out_channels, 1).to(device)
    
    def reset_internal_grads(self):
        grad = self.final_conv.weight.grad.clone().detach()
        self.zero_grad()
        self.final_conv.weight.grad = grad


def unet3d_ddpm(in_channels, num_classes, initialization_weights='kaiming', timestamp_diffusion=1000):
    model = DDPMUNet(in_channels, num_classes, timestamp_diffusion)
    init_weights(model, initialization_weights)
    
    return model



if __name__ == '__main__':
    model = DDPMUNet(3, 2)
    model.eval()
    input = torch.rand(2, 3, 128, 128)
    output, reconstr = model(input)
    # print(output)
    print(output.shape)
    print(reconstr.shape)

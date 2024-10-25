import torch
from torch import nn
from torch.nn import init

from denoising_diffusion_pytorch import Unet


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class DDPM_Wrapper(Unet):

    def forward(self, x, time, x_self_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)

        self.latent = x

        return self.final_conv(x)


class DDPMUNet(nn.Module):

    def __init__(self, in_chns, class_num, timestamp_diffusion=1000):
        super(DDPMUNet, self).__init__()

        self.net = DDPM_Wrapper(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            flash_attn = False,
            out_dim = in_chns,
        )

        self.timestamp_diffusion = timestamp_diffusion

        self.out_conv = nn.Conv2d(64, class_num,
                            kernel_size=3, padding=1)
    
    def forward(self, x):
        t = torch.randint(0, self.timestamp_diffusion, (x.shape[0],), device=x.device).long()
    
        reconstr = self.net(x, t)
        output = self.out_conv(self.net.latent)
        
        return output, reconstr
    
    def state_dict(self):
        return self.net.state_dict()
    
    def load_state_dict(self, state_dict, strict = ...):
        return self.net.load_state_dict(state_dict, strict)
    
    def reset_clf(self, out_channels):
        device = self.net.final_conv.weight.device
        self.net.final_conv = nn.Conv2d(self.net.final_conv.weight.shape[1], out_channels, 1).to(device)
        
    def reset_internal_grads(self):
        grad = self.out_conv.weight.grad.clone().detach()
        self.zero_grad()
        self.out_conv.weight.grad = grad



def unet_ddpm(in_channels, num_classes, initialization_weights='kaiming', timestamp_diffusion=1000):
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
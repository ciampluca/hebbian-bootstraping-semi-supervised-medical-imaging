import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


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


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dropout=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    

class ConvBlockLeakyRelu(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlockLeakyRelu, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    

class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlockLeakyRelu(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]
    
class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlockLeakyRelu(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)
    

class DecoderLeaky(nn.Module):
    def __init__(self, params):
        super(DecoderLeaky, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        # self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
        #                           kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        output = self.up4(x, x0)
        #output = self.out_conv(x)
        return output

    

class UNet_Transposed_Leaky(nn.Module):
    
    def __init__(self, in_chns, class_num, test_unsup_low_layer=False, linear_probe=True, multiple_layers=False):
        super(UNet_Transposed_Leaky, self).__init__()
        
        self.test_unsup_low_layer = test_unsup_low_layer

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        
        self.encoder = Encoder(params)
        self.main_decoder = DecoderLeaky(params)
        # self.aux_decoder1 = Decoder(params)
        # self.aux_decoder2 = Decoder(params)
        # self.aux_decoder3 = Decoder(params)
        
        if linear_probe:
            kernel_dim = 1
            padding = 0
            if not multiple_layers:
                self.out_conv = nn.Conv2d(params['feature_chns'][0], class_num,
                                kernel_size=kernel_dim, padding=padding)
            else:
                self.out_conv = nn.Sequential(
                    nn.Conv2d(params['feature_chns'][0], params['feature_chns'][0]*4, kernel_size=kernel_dim, padding=padding),
                    nn.ReLU(),
                    nn.Dropout(),
                    # nn.Conv2d(params['feature_chns'][0]*4, params['feature_chns'][0]*2, kernel_size=kernel_dim, padding=padding),
                    # nn.ReLU(),
                    # nn.Dropout(),
                    nn.Conv2d(params['feature_chns'][0]*4, class_num, kernel_size=kernel_dim, padding=padding)
                )
        else:
            kernel_dim = 3
            padding = 1
            self.out_conv = nn.Conv2d(params['feature_chns'][0], class_num,
                    kernel_size=kernel_dim, padding=padding)
        
        
        self.out_superpix = nn.Conv2d(params['feature_chns'][0], 2, kernel_size=1)

    def forward(self, x):
        feature = self.encoder(x)     
        
        dec_out = self.main_decoder(feature)
        main_seg = self.out_conv(dec_out)
        out_superpix = self.out_superpix(dec_out)
        # aux1_feature = [FeatureNoise()(i) for i in feature]
        # aux_seg1 = self.aux_decoder1(aux1_feature)
        # aux2_feature = [Dropout(i) for i in feature]
        # aux_seg2 = self.aux_decoder2(aux2_feature)
        # aux3_feature = [FeatureDropout(i) for i in feature]
        # aux_seg3 = self.aux_decoder3(aux3_feature)
        # return main_seg, aux_seg1, aux_seg2, aux_seg3
        
        return main_seg, out_superpix
    
    def reset_internal_grads(self):
        grad_dict = {}
        for n, p in self.out_conv.named_parameters():
            grad_dict[n] = p.grad.clone().detach()
            
        self.zero_grad()
        
        for n, p in self.out_conv.named_parameters():
            p.grad = grad_dict[n]


def unet_superpix(in_channels, num_classes, initialization_weights='kaiming'):
    model = UNet_Transposed_Leaky(in_channels, num_classes)
    init_weights(model, initialization_weights)
    
    return model



if __name__ == '__main__':
    model = UNet_Transposed_Leaky(3, 2)
    model.eval()
    input = torch.rand(2, 3, 128, 128)
    output = model(input)
    output = output[0].data.cpu().numpy()
    # print(output)
    print(output.shape)
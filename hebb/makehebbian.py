import torch
import torch.nn as nn
from .hebb import HebbianConv2d, HebbianConvTranspose2d


pad_layers = {'zeros': nn.ZeroPad2d, 'reflect': nn.ReflectionPad2d, 'replicate': nn.ReplicationPad2d,'circular': nn.CircularPad2d if hasattr(nn, 'CircularPad2d') else None}
default_hebb_params = {'w_nrm': True, 'act': nn.Identity(), 'mode': HebbianConvTranspose2d.MODE_SWTA_T, 'k': 50, 'patchwise': True, 'contrast': 1., 'uniformity': False, 'alpha': 0.}

class UnsqueezeLast(nn.Module):
    def __init__(self, d=2):
        super().__init__()
        self.d = d
    
    def forward(self, x):
        return x.reshape(*x.shape, *([1]*self.d))

class FlattenLast(nn.Module):
    def __init__(self, d=2):
        super().__init__()
        self.d = d
    
    def forward(self, x):
        return x.reshape(*(x.shape[:-self.d-1]), -1)

def adjust_hebbian_params(hebb_params):
    adj_hebb_params = hebb_params.copy()
    mode = adj_hebb_params.get('mode', None)
    if mode is None: return adj_hebb_params
    if mode.endswith('_t'): adj_hebb_params['mode'] = mode[:-2]
    return adj_hebb_params

def init_weights(m, init_type='normal', gain=0.02):
    if init_type == 'normal':
        nn.init.normal_(m.weight.data, 0.0, gain)
    elif init_type == 'xavier':
        nn.init.xavier_normal_(m.weight.data, gain=gain)
    elif init_type == 'kaiming':
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif init_type == 'orthogonal':
        nn.init.orthogonal_(m.weight.data, gain=gain)
    else:
        raise NotImplementedError("Unsupported initialization method {}".format(init_type))
    return m

def makehebbian(model, exclude=None, hebb_params=None):
    if hebb_params is None: hebb_params = default_hebb_params
    
    def _match(n, e):
        #return e.endswith(n)
        return e == n
    
    if exclude is None: exclude = []
    exclude = [(n, m) for n, m in model.named_modules() if any([_match(n, e) for e in exclude])]
    print("Layers excluded from conversion to Hebbian: {}".format([n for n, m in exclude]))
    exclude = [m for n, p in exclude for m in [*p.modules()]]
    
    def _makehebbian(module):
        for n, m in module.named_children():
            if m in exclude: continue
            if type(m) is nn.Conv2d:
                if m.dilation != 1 and m.dilation != (1, 1): raise RuntimeError("Dilation not supported with Hebbian layers")
                if m.groups != 1: raise RuntimeError("Grouped convolution not supported with Hebbian layers")
                PadLayer = pad_layers.get(m.padding_mode, None)
                if PadLayer is None: raise RuntimeError("Padding mode {} unsupported".format(m.padding_mode))
                setattr(module, n, nn.Sequential(
                    PadLayer(m.padding if isinstance(m.padding, int) else (m.padding[0], m.padding[0], m.padding[1], m.padding[1])),
                    init_weights(HebbianConv2d(m.in_channels, m.out_channels, m.kernel_size, m.stride, **adjust_hebbian_params(hebb_params))),
                ))
            elif type(m) is nn.ConvTranspose2d:
                if m.dilation != 1 and m.dilation != (1, 1): raise RuntimeError("Dilation not supported with Hebbian layers")
                if m.groups != 1: raise RuntimeError("Grouped convolution not supported with Hebbian layers")
                PadLayer = pad_layers.get(m.padding_mode, None)
                if PadLayer is None: raise RuntimeError("Padding mode {} unsupported".format(m.padding_mode))
                setattr(module, n, nn.Sequential(
                    PadLayer(m.padding if isinstance(m.padding, int) else (m.padding[0], m.padding[0], m.padding[1], m.padding[1])),
                    init_weights(HebbianConvTranspose2d(m.in_channels, m.out_channels, m.kernel_size, m.stride, **hebb_params)),
                    PadLayer((0, m.output_padding, 0, m.output_padding) if isinstance(m.output_padding, int) else (0, m.output_padding[0], 0, m.output_padding[1])),
                ))
            elif type(m) is nn.Linear:
                setattr(module, n, nn.Sequential(
                    UnsqueezeLast(2),
                    init_weights(HebbianConv2d(m.in_features, m.out_features, 1, 1, **adjust_hebbian_params(hebb_params))),
                    FlattenLast(2),
                ))
            else:
                for p in m.parameters(): p.requires_grad = False
    
    model.apply(_makehebbian)


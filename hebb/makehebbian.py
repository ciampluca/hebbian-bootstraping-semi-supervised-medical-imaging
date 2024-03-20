import torch
import torch.nn as nn
from .hebb import HebbianConv2d, HebbianConvTranspose2d
from .hebb3d import HebbianConv3d, HebbianConvTranspose3d


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
    
    def _replacelayer(m, n, l):
        #setattr(m, n, l)
        m.register_module(n, l)
    
    def _makehebbian(module):
        for n, m in module.named_children():
            if m in exclude: continue
            if type(m) is nn.Conv2d:
                if m.dilation != 1 and m.dilation != (1, 1): raise RuntimeError("Dilation not supported with Hebbian layers")
                if m.groups != 1: raise RuntimeError("Grouped convolution not supported with Hebbian layers")
                _replacelayer(module, n, init_weights(HebbianConv2d(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding, False, **adjust_hebbian_params(hebb_params)), init_type='kaiming'))
            elif type(m) is nn.ConvTranspose2d:
                if m.dilation != 1 and m.dilation != (1, 1): raise RuntimeError("Dilation not supported with Hebbian layers")
                if m.groups != 1: raise RuntimeError("Grouped convolution not supported with Hebbian layers")
                _replacelayer(module, n, init_weights(HebbianConvTranspose2d(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding, False, **hebb_params), init_type='kaiming'))
            if type(m) is nn.Conv3d:
                if m.dilation != 1 and m.dilation != (1, 1, 1): raise RuntimeError("Dilation not supported with Hebbian layers")
                if m.groups != 1: raise RuntimeError("Grouped convolution not supported with Hebbian layers")
                _replacelayer(module, n, init_weights(HebbianConv3d(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding, False, **adjust_hebbian_params(hebb_params)), init_type='kaiming'))
            elif type(m) is nn.ConvTranspose3d:
                if m.dilation != 1 and m.dilation != (1, 1, 1): raise RuntimeError("Dilation not supported with Hebbian layers")
                if m.groups != 1: raise RuntimeError("Grouped convolution not supported with Hebbian layers")
                _replacelayer(module, n, init_weights(HebbianConvTranspose3d(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding, False, **hebb_params), init_type='kaiming'))
            elif type(m) is nn.Linear:
                _replacelayer(module, n, nn.Sequential(UnsqueezeLast(2), init_weights(HebbianConv2d(m.in_features, m.out_features, 1, 1, **adjust_hebbian_params(hebb_params)), init_type='kaiming'), FlattenLast(2)))
            else:
                for p in m.parameters(recurse=False): p.requires_grad = False
    
    model.apply(_makehebbian)


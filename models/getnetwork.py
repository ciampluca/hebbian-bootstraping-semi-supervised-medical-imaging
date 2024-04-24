import sys
from models import *

def get_network(network, in_channels, num_classes, **kwargs):

    # 2d networks
    if network == 'unet':
        net = unet(in_channels, num_classes)
    elif network == 'unet_urpc':
        net = unet_urpc(in_channels, num_classes)
    elif network == 'unet_cct':
        net = unet_cct(in_channels, num_classes)

    # 3d networks
    if network == 'unet3d':
        net = unet3d(in_channels, num_classes)
    elif network == 'unet3d_min':
        net = unet3d_min(in_channels, num_classes)
    elif network == 'unet3d_urpc':
        net = unet3d_urpc(in_channels, num_classes)
    elif network == 'unet3d_cct':
        net = unet3d_cct(in_channels, num_classes)
    elif network == 'unet3d_cct_min':
        net = unet3d_cct_min(in_channels, num_classes)
    elif network == 'unet3d_dtc':
        net = unet3d_dtc(in_channels, num_classes)
    elif network == 'vnet':
        net = vnet(in_channels, num_classes)
    elif network == 'vnet_cct':
        net = vnet_cct(in_channels, num_classes)
    elif network == 'vnet_dtc':
        net = vnet_dtc(in_channels, num_classes)
    else:
        print('the network you have entered is not supported yet')
        sys.exit()
    return net

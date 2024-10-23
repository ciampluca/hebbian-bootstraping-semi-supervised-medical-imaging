import sys
from models import *

def get_network(network, in_channels, num_classes, init_weights='kaiming', **kwargs):

    # 2d networks
    print(network)
    if network == 'unet':
        net = unet(in_channels, num_classes, init_weights)
    elif network == 'unet_urpc':
        net = unet_urpc(in_channels, num_classes, init_weights)
    elif network == 'unet_cct':
        net = unet_cct(in_channels, num_classes, init_weights)
    elif network == "unet_vae":
        net = unet_vae(in_channels, num_classes, init_weights)
    elif network == "unet_superpix":
        net = unet_superpix(in_channels, num_classes, init_weights)

    # 3d networks
    elif network == 'unet3d':
        net = unet3d(in_channels, num_classes, init_weights)
    elif network == 'unet3d_urpc':
        net = unet3d_urpc(in_channels, num_classes, init_weights)
    elif network == 'unet3d_cct':
        net = unet3d_cct(in_channels, num_classes, init_weights)
    elif network == 'unet3d_dtc':
        net = unet3d_dtc(in_channels, num_classes, init_weights)
    elif network == 'vnet':
        net = vnet(in_channels, num_classes, init_weights)
    else:
        print('the network you have entered is not supported yet')
        sys.exit()
    return net

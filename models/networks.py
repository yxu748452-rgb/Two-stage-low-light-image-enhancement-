import torch
#import models.archs.discriminator_vgg_arch as SRGAN_arch

import models.archs.FFTLLIE as FFTLLIE


# from models.archs
# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    # video restoration
    if which_model == 'Net':
        netG = FFTLLIE.FFTLLIE(nc=opt_net['nc'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG


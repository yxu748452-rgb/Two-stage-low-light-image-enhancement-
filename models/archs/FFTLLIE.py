import functools
from models.archs.SFBlock import *
import kornia
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
###############################
class FFTLLIE(nn.Module):
    def __init__(self, nc=8):
        super(FFTLLIE, self).__init__()

        self.conv0 = nn.Conv2d(3, nc, 3, 1, 1)
        #self.conv1 = nn.Conv2d(3, 3, 3, 1, 1)
        # AMPLITUDE ENHANCEMENT
        self.contrastNet = ContrastProcess(3,16)
        self.AmpNet = AmpenhanBlock1(nc)
        self.PhaNet = PhaseNet(nc)
        self.out1 = nn.Conv2d(nc, 3, 3, 1, 1)
        self.convoutfinal = nn.Conv2d(3, 3, 1, 1, 0)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # AMPLITUDE ENHANCEMENT
        x_1=x
        _, _, H, W = x.shape
        image_fft = torch.fft.fft2(x, norm='backward')
        amp_image = torch.abs(image_fft)
        pha_image = torch.angle(image_fft)

        adjust_amp = self.AmpNet(self.conv0(x))        #调整图
        # _, _, H, W = amp_image1.shape
        # image_fft1 = torch.fft.fft2(amp_image1, norm='backward')
        # amp_image = torch.abs(image_fft1)
        amp_image = amp_image * adjust_amp

        real_image_enhanced = amp_image * torch.cos(pha_image)
        imag_image_enhanced = amp_image * torch.sin(pha_image)
        img_amp_enhanced = torch.fft.ifft2(torch.complex(real_image_enhanced, imag_image_enhanced), s=(H, W),
                                           norm='backward').real

        #img_amp_enhanced = self.sig(self.out1(img_amp_enhanced))
        x_out1 = img_amp_enhanced  # 第一阶段的输出
        x_contrast = self.contrastNet(x_out1,x_1)
        x_out2 = self.PhaNet(x_contrast)

        return x_out1,x_out2,amp_image


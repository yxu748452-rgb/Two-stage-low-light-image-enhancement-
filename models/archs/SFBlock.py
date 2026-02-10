import torch
import torch.nn as nn
import torch.nn.functional as F

##空间卷积
class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(nc,nc,3,1,1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return x+self.block(x)



class AmpenhanBlock(nn.Module):
    def __init__(self, nc):
        super(AmpenhanBlock, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(nc,nc,1,1,0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc,nc,1,1,0))
        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.conv2 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.conv3 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.conv4 = nn.Sequential(
            nn.Conv2d(nc * 2, nc * 2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc * 2, nc * 2, 1, 1, 0),
            nn.Conv2d(nc * 2, nc, 1, 1, 0),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(nc * 2, nc * 2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc * 2, nc * 2, 1, 1, 0),
            nn.Conv2d(nc * 2, nc, 1, 1, 0),
        )
        self.convout = nn.Sequential(
            nn.Conv2d(nc * 2, nc * 2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc * 2, nc, 1, 1, 0),
        )

    def forward(self,x):
        x_out_0 = self.conv0(x)
        x_out_1 = self.conv1(x_out_0)
        x_out_2 = self.conv2(x_out_1)
        x_out_3 = self.conv3(x_out_2)
        x_out_4 = self.conv4(torch.cat((x_out_2, x_out_3), dim=1))
        x_out_5 = self.conv5(torch.cat((x_out_1, x_out_4), dim=1))
        x_out = self.convout(torch.cat((x_out_0, x_out_5), dim=1))

        return x_out





class FuseBlock(nn.Module):
    def __init__(self, channels):
        super(FuseBlock, self).__init__()
        self.fre = nn.Conv2d(channels, channels, 3, 1, 1)
        self.spa = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fuse = nn.Sequential(nn.Conv2d(2*channels, channels, 3, 1, 1), nn.Conv2d(channels, 2*channels, 3, 1, 1), nn.Sigmoid())
    def forward(self, spa, fre):
        ori = spa
        fre = self.fre(fre)+fre
        spa = self.spa(spa)+spa
        fuse = self.fuse(torch.cat((fre, spa), 1))
        fre_a, spa_a = fuse.chunk(2, dim=1)
        spa = spa_a * spa
        fre = fre * fre_a
        res = fre + spa
        res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res

class ResBlock_fft_bench(nn.Module):
    def __init__(self, n_feat):
        super(ResBlock_fft_bench, self).__init__()
        self.fpre = nn.Conv2d(n_feat, n_feat, 1, 1, 0)
        self.main = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1)
        self.mag = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0)
        self.pha = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
        )
    def forward(self, x):
        _, _, H, W = x.shape
        fre = torch.fft.rfft2(self.fpre(x), norm='backward')
        mag = torch.abs(fre)
        pha = torch.angle(fre)
        mag_out = self.mag(mag)
        mag_res = mag_out - mag
        pooling = torch.nn.functional.adaptive_avg_pool2d(mag_res, (1, 1))
        pooling = torch.nn.functional.softmax(pooling, dim=1)
        pha1 = pha * pooling
        pha1 = self.pha(pha1)
        pha_out = pha1 + pha
        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        fre_out = torch.complex(real, imag)
        y = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')

        return self.main(x) + y


class ResBlock_fft_bench1(nn.Module):
    def __init__(self, n_feat):
        super(ResBlock_fft_bench1, self).__init__()
        self.fpre = nn.Conv2d(n_feat, n_feat, 1, 1, 0)
        self.main = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1)
        #self.mag = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0)
        self.conv = nn.Conv2d(n_feat*2, n_feat, kernel_size=1, padding=0)
        self.mag = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
        )
        self.pha = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
        )
    def forward(self, x):
        _, _, H, W = x.shape
        fre = torch.fft.rfft2(self.fpre(x), norm='backward')
        mag = torch.abs(fre)
        pha = torch.angle(fre)
        mag_out = self.mag(mag)
        mag_res = mag_out - mag
        pooling_avg = torch.nn.functional.adaptive_avg_pool2d(mag_res, (1, 1))
        pooling_max = torch.nn.functional.adaptive_max_pool2d(mag_res, (1, 1))
        pooling = torch.cat((pooling_avg,pooling_max),1)
        pooling = self.conv(pooling)
        pooling = torch.nn.functional.softmax(pooling, dim=1)
        pha1 = pha * pooling
        pha1 = self.pha(pha1)
        pha_out = pha1 + pha
        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        fre_out = torch.complex(real, imag)
        y = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')

        return self.main(x) + y




class ProcessBlock_1(nn.Module):
    def __init__(self, nc):
        super(ProcessBlock_1,self).__init__()
        self.spatial_process = SpaBlock(nc)
        self.frequency_process = ResBlock_fft_bench(nc)
        self.fuse_process = FuseBlock(nc)
    def forward(self, x):
        xori = x
        x_freq = self.frequency_process(x)
        x_spatial = self.spatial_process(x)
        x_fuseout = self.fuse_process(x_spatial,x_freq)

        return x_freq,x_fuseout

class ProcessBlock(nn.Module):
    def __init__(self, nc):
        super(ProcessBlock,self).__init__()
        self.spatial_process = SpaBlock(nc)
        self.frequency_process = ResBlock_fft_bench(nc)
        self.fuse_process = FuseBlock(nc)
    def forward(self, x_F,x_S):
        #xori = x
        x_freq = self.frequency_process(x_F)
        x_spatial = self.spatial_process(x_S)
        x_fuseout = self.fuse_process(x_spatial,x_freq)

        return x_F,x_fuseout
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#第二阶段 输入channel：3
class PhaseNet(nn.Module):
    def __init__(self, nc):
        super(PhaseNet,self).__init__()
        self.conv0 = nn.Conv2d(3,nc,1,1,0)
        self.conv1 = ProcessBlock_1(nc)
        self.downsample1 = nn.Conv2d(nc,nc*2,stride=2,kernel_size=2,padding=0)
        self.downsample1_1 = nn.Conv2d(nc, nc*2, stride=2, kernel_size=2, padding=0)

        self.conv2 = ProcessBlock(nc*2)
        self.downsample2 = nn.Conv2d(nc*2,nc*3,stride=2,kernel_size=2,padding=0)
        self.downsample2_2 = nn.Conv2d(nc*2, nc*3, stride=2, kernel_size=2, padding=0)

        self.conv3 = ProcessBlock(nc*3)
        self.up1 = nn.ConvTranspose2d(nc*5,nc*2,1,1)
        self.up1_1 = nn.ConvTranspose2d(nc * 5, nc * 2, 1, 1)

        self.conv4 = ProcessBlock(nc*2)
        self.up2 = nn.ConvTranspose2d(nc*3,nc*1,1,1)
        self.up2_2 = nn.ConvTranspose2d(nc * 3, nc * 1, 1, 1)

        self.conv5 = ProcessBlock(nc)
        self.convout = nn.Conv2d(nc*2,3,1,1,0)
        self.convoutfinal = nn.Conv2d(3, 3, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()
        self.pro = nn.Conv2d(3, 3, 1, 1, 0)

    def forward(self, x):
        x_ori = x
        #x = self.conv0(x)

        x_fre1,x_fuse1 = self.conv1(x)
        x_fre1o = self.downsample1(x_fre1)
        x_fuse1o = self.downsample1(x_fuse1)

        x_fre2, x_fuse2 = self.conv2(x_fre1o,x_fuse1o)
        x_fre2o = self.downsample2(x_fre2)
        x_fuse2o = self.downsample2(x_fuse2)

        x_fre3, x_fuse3 = self.conv3(x_fre2o, x_fuse2o)

        x_fre3 = self.up1(torch.cat([F.interpolate(x_fre3,size=(x_fre2.size()[2],x_fre2.size()[3]),mode='bilinear'),x_fre2],1))
        x_fuse3 = self.up1(torch.cat([F.interpolate(x_fuse3, size=(x_fuse2.size()[2], x_fuse2.size()[3]), mode='bilinear'), x_fuse2], 1))

        x_fre4, x_fuse4 = self.conv4(x_fre3,x_fuse3)
        x_fre4 = self.up2(torch.cat([F.interpolate(x_fre4,size=(x_fre1.size()[2],x_fre1.size()[3]),mode='bilinear'),x_fre1],1))
        x_fuse4 = self.up2_2(torch.cat([F.interpolate(x_fuse4, size=(x_fuse1.size()[2], x_fuse1.size()[3]), mode='bilinear'), x_fuse1], 1))

        x_fre5, x_fuse5 = self.conv5(x_fre4, x_fuse4)
        xout = self.convout(torch.cat((x_fre5,x_fuse5),1))
        #xout = x_ori + xout
        xfinal = self.convoutfinal(xout)

        return xfinal


#在一阶段 使用幅度和相位为卷积
class AmpenhanBlock1(nn.Module):
    def __init__(self, nc):
        super(AmpenhanBlock1, self).__init__()
        self.conv0 = ProcessBlock1(nc)
        self.conv1 = ProcessBlock1(nc)
        self.conv2 = ProcessBlock1(nc)
        self.conv3 = ProcessBlock1(nc)

        self.conv4 = nn.Sequential(
                        ProcessBlock1(nc*2),
                      nn.Conv2d(nc*2,nc,3,1,1), )

        self.conv5 = nn.Sequential(
            ProcessBlock1(nc * 2),
            nn.Conv2d(nc * 2, nc, 3, 1, 1), )


        self.convout = nn.Sequential(
            ProcessBlock1(nc * 2),
            nn.Conv2d(nc*2,3,3,1,1))
        self.sigmoid= nn.Sigmoid()

    def forward(self,x):
        x_out_0 = self.conv0(x)
        x_out_1 = self.conv1(x_out_0)
        x_out_2 = self.conv2(x_out_1)
        x_out_3 = self.conv3(x_out_2)
        x_out_4 = self.conv4(torch.cat((x_out_2, x_out_3), dim=1))
        x_out_5 = self.conv5(torch.cat((x_out_1, x_out_4), dim=1))
        x_out = self.convout(torch.cat((x_out_0, x_out_5), dim=1))
        #x_out = self.sigmoid(x_out)
        return x_out


class ProcessBlock1(nn.Module):
    def __init__(self, nc):
        super(ProcessBlock1,self).__init__()
        self.spatial_process = SpaBlock(nc)
        self.frequency_process = ResBlock_fft_bench(nc)
        self.conv_out = nn.Conv2d(nc*2,nc,1,1,1,0)
    def forward(self, x):
        xori = x
        x_freq = self.frequency_process(x)
        #x_spatial = self.spatial_process(x)
        #x_out = torch.cat((x_freq,x_spatial),1)
        #x_out = self.conv_out(x_out)
        x_out = x_freq
        return x_out+xori


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class ContrastProcess(nn.Module):
    def __init__(self, in_nc,out_nc):
        super(ContrastProcess,self).__init__()
        self.cat = nn.Conv2d(2*in_nc,out_nc,1,1,0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.process = nn.Sequential(nn.Conv2d(in_nc * 2, in_nc // 2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(in_nc // 2, in_nc * 2, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
    def forward(self,x_amp,x):

        xcat = torch.cat([x_amp, x], 1)
        xcat = self.process(self.contrast(xcat) + self.avgpool(xcat)) * xcat
        x_out = self.cat(xcat)
        return x_out
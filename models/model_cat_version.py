from __future__ import print_function, division
import os
import pathlib
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import functional as TF
from matplotlib import pyplot as plt
from torchsummary import summary


# Number of workers for dataloader
workers = 2
# Batch size during training
# batch_size = 128
batch_size = 32
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 256

# Number of training epochs
num_epochs = 1000
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Each image is 256x256 in size
IMG_WIDTH = 128
IMG_HEIGHT = 128
INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 1

class downsample(nn.Module):
    def __init__(self, in_nc, out_nc, apply_norm=None):
        super(downsample, self).__init__()
        self.ngpu = ngpu
        conv =  nn.Conv2d(in_nc, out_nc, kernel_size=4,
                          stride=2, padding=1, bias=False)
        relu = nn.ReLU()

        if apply_norm == 'batch':
            norm = nn.BatchNorm2d(out_nc)
            self.model = nn.Sequential(*[conv, norm, relu])
        elif apply_norm == 'instance':
            norm = nn.InstanceNorm2d(out_nc)
            self.model = nn.Sequential(*[conv, norm, relu])
        else:
            self.model = nn.Sequential(*[conv, relu])

    def forward(self, x):
        return self.model(x)


class upsample(nn.Module):
    def __init__(self, in_nc, out_nc, apply_norm=None):
        super(upsample, self).__init__()
        self.ngpu = ngpu
        tconv = nn.ConvTranspose2d(in_nc, out_nc, kernel_size=4,
                                   stride=2, padding=1, bias=False)

        relu = nn.ReLU()
        if apply_norm == 'batch':
            norm = nn.BatchNorm2d(out_nc)
            self.model = nn.Sequential(*[tconv, norm, relu])
        elif apply_norm == 'instance':
            norm = nn.InstanceNorm2d(out_nc)
            self.model = nn.Sequential(*[tconv, norm, relu])
        else:
            self.model = nn.Sequential(*[tconv, relu])

    def forward(self, x):
        return self.model(x)

class flatconv(nn.Module):
    def __init__(self, in_nc, out_nc, apply_norm=None):
        super(flatconv, self).__init__()
        self.ngpu = ngpu
        fconv = nn.Conv2d(in_nc, out_nc, kernel_size=3,
                          stride=1, padding=1, bias=False)

        relu = nn.ReLU()
        if apply_norm == 'batch':
            norm = nn.BatchNorm2d(out_nc)
            self.model = nn.Sequential(*[fconv, norm, relu])
        elif apply_norm == 'instance':
            norm = nn.InstanceNorm2d(out_nc)
            self.model = nn.Sequential(*[fconv, norm, relu])
        else:
            self.model = nn.Sequential(*[fconv, relu])

    def forward(self, x):
        return self.model(x)

class first_flatconv(nn.Module):
    def __init__(self, in_nc, out_nc, apply_norm=None):
        super(first_flatconv, self).__init__()
        self.ngpu = ngpu
        fconv = nn.Conv2d(in_nc, out_nc, kernel_size=7,
                          stride=1, padding=3, bias=False, padding_mode='replicate')

        relu = nn.ReLU()
        if apply_norm == 'batch':
            norm = nn.BatchNorm2d(out_nc)
            self.model = nn.Sequential(*[fconv, norm, relu])
        elif apply_norm == 'instance':
            norm = nn.InstanceNorm2d(out_nc)
            self.model = nn.Sequential(*[fconv, norm, relu])
        else:
            self.model = nn.Sequential(*[fconv, relu])

    def forward(self, x):
        return self.model(x)

class ResNet(nn.Module):
    def __init__(self, dim, padding_mode='replicate', apply_norm=None):
        super(ResNet, self).__init__()
        self.ngpu = ngpu
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                          padding=1, padding_mode=padding_mode, bias=False)
        self.norm1 = nn.BatchNorm2d(dim)
        if apply_norm == 'batch':
            self.norm1 = nn.BatchNorm2d(dim)
        elif apply_norm == 'instance':
            self.norm1 = nn.InstanceNorm2d(dim)
        else:
            self.norm1 = nn.Identity()
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                          padding=1, padding_mode=padding_mode, bias=False)
        if apply_norm == 'batch':
            self.norm2 = nn.BatchNorm2d(dim)
        elif apply_norm == 'instance':
            self.norm2 = nn.InstanceNorm2d(dim)
        else:
            self.norm2= nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(x)

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        out = out + x
        out = self.relu(out)

        return out

class Generator(nn.Module):
    def __init__(self, in_nc=4, out_nc=1, ngf=64, apply_norm=None):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        # input (batch_size, in_nc, 128, 128)
        # self.fconv_d1 = flatconv(in_nc, ngf)
        self.fconv_d1 = first_flatconv(in_nc, ngf)
        self.fconv_d2 = flatconv(ngf, ngf)

        # size (batch_size, ngf, 128, 128)
        self.down1 = downsample(ngf, ngf*2)
        self.fconv_d3 = flatconv(ngf*2, ngf*2)

        # size (batch_size, ngf*2, 64, 64)
        self.down2 = downsample(ngf*2, ngf*4)
        self.resSeq = nn.Sequential(*[ResNet(ngf*4) for i in range(9)])

        # size (batch_size, ngf*4, 32, 32)
        self.up2 = upsample(ngf*4, ngf*2)
        self.fconv_u3 = flatconv(ngf*2, ngf*2)

        # size (batch_size, ngf*2 + ngf*2(skip), 64, 64)
        self.up1 = upsample(ngf*2 + ngf*2, ngf*2)
        self.fconv_u2 = flatconv(ngf*2, ngf*2)
        self.fconv_u1 = flatconv(ngf*2, ngf)

        # size (batch_size, ngf + ngf(skip), 128, 128)->(batch_size, out_nc, 128, 128)
        self.lastConvLayer = nn.Sequential(
            nn.Conv2d(ngf*2, out_nc, kernel_size=3,stride=1,
                               padding=1, padding_mode='replicate', bias=False),
            nn.Tanh())

    def forward(self, inputs):

        out_d1 = self.fconv_d1(inputs)
        out_d2 = self.fconv_d2(out_d1)
        down1  = self.down1(out_d2)
        out_d3 = self.fconv_d3(down1)
        down2  = self.down2(out_d3)

        resnet = self.resSeq(down2)

        up2    = self.up2(resnet)
        out_u3 = self.fconv_u3(up2)
        up1    = self.up1(torch.cat([out_u3, out_d3], 1))        #skip connection
        out_u2 = self.fconv_u2(up1)
        out_u1 = self.fconv_u1(out_u2)
        out    = self.lastConvLayer(torch.cat([out_u1, out_d1], 1))        #skip connection

        return out


class disc_downsample(nn.Module):
    def __init__(self, in_nc, out_nc, apply_norm=None):
        super(disc_downsample, self).__init__()
        self.ngpu = ngpu
        conv =  nn.Conv2d(in_nc, out_nc, kernel_size=4,
                          stride=2, padding=1, bias=False)
        lrelu = nn.LeakyReLU(0.2)

        if apply_norm == 'batch':
            norm = nn.BatchNorm2d(out_nc)
            self.model = nn.Sequential(*[conv, norm, lrelu])
        if apply_norm == 'instance':
            norm = nn.InstanceNorm2d(out_nc)
            self.model = nn.Sequential(*[conv, norm, lrelu])
        else:
            self.model = nn.Sequential(*[conv, lrelu])

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_ch=1, out_nc=1, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        # input (batch_size, in_ch, 128, 128)
        self.down1 = disc_downsample(in_ch, ndf)
        #  (batch_size, ndf, 64, 64)
        self.down2 = disc_downsample(ndf, ndf*2)
        #  (batch_size, ndf*2, 32, 32)
        self.down3 = disc_downsample(ndf*2, ndf*4)
        #  (batch_size, ndf*4, 16, 16)
        self.down4 = disc_downsample(ndf*4, ndf*8)
        #  (batch_size, ndf*8, 8, 8)
        self.down5 = disc_downsample(ndf*8, ndf*8)
        #  (batch_size, ndf*8, 4, 4)
        self.down6 = disc_downsample(ndf*8, ndf*8)
        #  (batch_size, ndf*8, 2, 2)
        self.down7 = disc_downsample(ndf*8, ndf*16)
        #  (batch_size, ndf*16, 1, 1)
        self.lastConvLayer = nn.Conv2d(ndf*16, out_nc, kernel_size=1,stride=1,
                               padding=1, padding_mode='replicate', bias=False)

    def forward(self, inputs):
        out_down1 = self.down1(inputs)
        out_down2 = self.down2(out_down1)
        out_down3 = self.down3(out_down2)
        out_down4 = self.down4(out_down3)
        out_down5 = self.down5(out_down4)
        out_down6 = self.down6(out_down5)
        out_down7 = self.down7(out_down6)
        out       = self.lastConvLayer(out_down7)

        return out


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=1, out_nc=1, ndf=64):
        super(PatchDiscriminator, self).__init__()
        self.ngpu = ngpu
        # input (batch_size, in_ch, 128, 128)
        self.down1 = disc_downsample(in_ch, ndf)
        #  (batch_size, ndf, 64, 64)
        self.down2 = disc_downsample(ndf, ndf*2)
        #  (batch_size, ndf*2, 32, 32)
        self.down3 = disc_downsample(ndf*2, ndf*4)
        #  (batch_size, ndf*4, 16, 16)
        self.conv1 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4,
                               stride=1, padding=1, bias=False)
        self.inorm = nn.InstanceNorm2d(ndf*8)
        self.lrelu = nn.LeakyReLU(0.2)
        self.lastConvLayer = nn.Conv2d(ndf*8, out_nc, kernel_size=1,stride=1,
                               padding=1, padding_mode='replicate', bias=False)

    def forward(self, inputs):
        out_down1 = self.down1(inputs)
        out_down2 = self.down2(out_down1)
        out_down3 = self.down3(out_down2)
        out_c0    = self.conv1(out_down3)
        out_bn0   = self.inorm(out_c0)
        out_lr0   = self.lrelu(out_bn0)

        out       = self.lastConvLayer(out_lr0)

        return out

if __name__ == '__main__':
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    generator = Generator(in_nc=3).to(device)
    discriminator = Discriminator().to(device)
    patch_discriminator = PatchDiscriminator().to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        generator = nn.DataParallel(generator, list(range(ngpu)))

    # Print the model
    summary(generator, (3, 128, 128))
    summary(discriminator, (1, 128, 128))
    summary(patch_discriminator, (1, 128, 128))
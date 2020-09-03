import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import CenterCrop, Padding
from layers import *
from prettytable import PrettyTable

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Unet(nn.Module):
    def __init__(self, input_dim=1, separable=False, method='upsample'):
        super().__init__()
        self.doubleconv = DoubleConv(1, 64, separable)
        self.down1 = Down(64, 128, separable)
        self.down2 = Down(128, 256, separable)
        self.down3 = Down(256, 512, separable)
        self.down4 = Down(512, 1024, separable)

        self.up1 = Up(1024, 512, separable, method)
        self.up2 = Up(512, 256, separable, method)
        self.up3 = Up(256, 128, separable, method)
        self.up4 = Up(128, 64, separable, method)

        self.outconv = nn.Conv2d(64, 1, 1)


    def forward(self, x):
        x1 = self.doubleconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outconv(x)

        return x

    def summary(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, separable):
        super().__init__()
        self.convs = DoubleConv(in_channels, out_channels, separab=separable)

    def forward(self, x):
        x = F.max_pool2d(x, 2)
        x = self.convs(x)
        return x



class Up(nn.Module):
    def __init__(self, in_channels, out_channels, separable, method='upsample'):
        super().__init__()
        self.convs = DoubleConv(in_channels, out_channels, separable=separable)
        if method == 'upsample':
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                PointWiseConv(in_channels, out_channels, bias=False)
            )
        elif method == 'transpose':
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2,
                                               bias=False)

        self.pad = Padding()

    def forward(self, bottom_x, skip_x):
        bottom_x = self.upsample(bottom_x) #[B, C, H, W]
        S = skip_x.size(3)
        bottom_x = self.pad(bottom_x, S)

        concate_x = torch.cat([skip_x, bottom_x], dim=1)
        return self.convs(concate_x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, separable):
        super().__init__()
        if separable is True:
            self.layers = nn.Sequential(
                DepthWiseSeparableConv(in_channels, out_channels, 3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                DepthWiseSeparableConv(out_channels, out_channels, 3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

    def forward(self, x):
        return self.layers(x)




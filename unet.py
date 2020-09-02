import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import CenterCrop, Padding

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Unet(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        self.doubleconv = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

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


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = F.max_pool2d(x, 2)
        x = self.convs(x)
        return x



class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = DoubleConv(in_channels, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ch_extract = nn.Linear(in_channels, out_channels)
        self.pad = Padding()

    def forward(self, bottom_x, skip_x):
        bottom_x = self.upsample(bottom_x) #[B, C, H, W]
        bottom_x = bottom_x.permute((0,2,3,1)) #[B, H, W, C]
        bottom_x = self.ch_extract(bottom_x).permute((0, 3, 1, 2)) #[B, C/2, H, W]
        # bottom_x < skip_x
        S = skip_x.size(3)
        bottom_x = self.pad(bottom_x, S)

        concate_x = torch.cat([skip_x, bottom_x], dim=1)
        return self.convs(concate_x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
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




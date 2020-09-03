import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

def conv1x1(in_channels, out_channels):
    return PointWiseConv(in_channels, out_channels, bias=False)

def conv3x3(in_channels, out_channels, dilation=1, separable=False):
    if separable == True:
        return DepthWiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=dilation,
                                      dilation=dilation, bias=False)
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation,
                     bias=False)

def batch_norm(channels):
    return nn.BatchNorm2d(channels)

def downsample2x(in_channels, out_channels, separable=False):
    if separable == True:
        return DepthWiseSeparableConv(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)

class BottleNeck(nn.Module):
    def __init__(self, in_channels, hidden_channels, expansion=4, separable=False, down=True):
        super().__init__()
        self.conv1 = conv1x1(in_channels, hidden_channels)
        self.bn1 = batch_norm(hidden_channels)
        self.conv2 = conv3x3(hidden_channels, hidden_channels, separable=separable)
        self.bn2 = batch_norm(hidden_channels)
        self.conv3 = conv1x1(hidden_channels, expansion * hidden_channels)
        self.bn3 = batch_norm(hidden_channels * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.down = down
        self.downsample = downsample2x(hidden_channels * expansion, hidden_channels * expansion, separable=separable)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down is True:
            out = self.downsample(out)
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, in_channels, separable=False):
        super(ResNet, self).__init__()




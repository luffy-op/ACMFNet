import numpy as np
import torch
from torch import nn
from torch.nn import init

from loguru import logger


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        # padding = 3 if kernel_size == 7 else 1
        # self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        logger.error(result.shape)
        output = self.conv(result)
        output = self.sigmoid(output)
        logger.error(output.shape)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.float()  ############################
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        # print(out.shape)
        # print(self.sa(out).shape)
        out = out * self.sa(out)
        return (out + residual).half()  ######################


class Conv(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2048, 2048, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


if __name__ == '__main__':
    # input = torch.randn(100, 256, 56, 56)
    # input = torch.randn(100, 512, 28, 28)
    # input = torch.randn(100, 1024, 14, 14)
    input = torch.randn(100, 2048, 7, 7)
    kernel_size = input.shape[2]
    # cbam = CBAMBlock(channel=2048, reduction=16, kernel_size=7)
    # output = cbam(input)
    aaaa = Conv(kernel_size=3)
    output = aaaa(input)
    print(output.shape)

# torch.Size([100, 512, 28, 28])
# torch.Size([100, 1024, 14, 14])
# torch.Size([100, 2048, 7, 7])

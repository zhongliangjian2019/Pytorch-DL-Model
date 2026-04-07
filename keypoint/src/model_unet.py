""" Parts of the U-Net model """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        pl = torch.div(diffX, 2, rounding_mode='trunc')
        pr = diffX - pl
        pt = torch.div(diffY, 2, rounding_mode='trunc')
        pd = diffY - pt
        x1 = F.pad(x1, [pl, pr, pt, pd])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.conv(x))
class UNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = False, is_eval: bool = False):
        super(UNet, self).__init__()
        self.is_eval = is_eval
        self.in_channel = n_channels
        self.num_classes = n_classes
        self.bilinear = bilinear
        scale = 8
        self.inc = (DoubleConv(n_channels, scale))
        self.down1 = (Down(scale, scale * 2))
        self.down2 = (Down(scale * 2, scale * 4))
        self.down3 = (Down(scale * 4, scale * 8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(scale * 8, scale * 16 // factor))
        self.up1 = (Up(scale * 16, scale * 8 // factor, bilinear))
        self.up2 = (Up(scale * 8, scale * 4 // factor, bilinear))
        self.up3 = (Up(scale * 4, scale * 2 // factor, bilinear))
        self.up4 = (Up(scale * 2, scale, bilinear))
        self.outc = (OutConv(scale, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

if __name__ == '__main__':
    net = UNet(n_channels=1, n_classes=1, bilinear=True)
    summary(net, (1, 1, 512, 512))
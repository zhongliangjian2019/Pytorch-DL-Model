"""
关键点检测模型：基于Unet构建, 思想来源于CenterNet
"""
import torch
import torch.nn as nn
from torchinfo import summary
from model_unet import UNet

class BackboneUnet(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(BackboneUnet, self).__init__()
        self.backbone = UNet(in_channel, num_classes, bilinear=True, is_eval=False)

    def forward(self, x):
        x1 = self.backbone.inc(x)
        x2 = self.backbone.down1(x1)
        x3 = self.backbone.down2(x2)
        x4 = self.backbone.down3(x3)
        x5 = self.backbone.down4(x4)
        x = self.backbone.up1(x5, x4)
        x = self.backbone.up2(x, x3)
        x = self.backbone.up3(x, x2)
        x = self.backbone.up4(x, x1)
        return x

class Neck(nn.Module):
    def __init__(self, in_channels: list, out_channel: int):
        super(Neck, self).__init__()
        self.layers = nn.ModuleList()
        for channel in in_channels:
            self.layers.append(nn.Sequential(
            nn.Conv2d(channel, out_channel,
                      kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            ))

    def forward(self, inputs):
        features = [layer(x) for x, layer in zip(inputs, self.layers)]
        return features

class Head(nn.Module):
    def __init__(self, num_classes=1, channel=64):
        super(Head, self).__init__()
        # 热力图预测部分
        self.cls_head = nn.Conv2d(channel, num_classes, kernel_size=1, stride=1, padding=0)
        # 中心点预测的部分
        self.reg_head = nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        hm = nn.functional.sigmoid(self.cls_head(x))
        offset = self.reg_head(x)
        return hm, offset

class KeyPointModel(nn.Module):
    """关键点检测模型"""
    def __init__(self, in_channel: int = 1, num_classes: int = 4, is_eval: bool = False):
        super(KeyPointModel, self).__init__()
        self.backbone = BackboneUnet(in_channel, num_classes)
        self.neck = Neck(in_channels=[8], out_channel=8)
        self.head = Head(num_classes=num_classes, channel=8)
        self.is_eval = is_eval


    def forward(self, x):
        x = self.backbone(x)
        x = self.neck([x])[0]
        hm, offset = self.head(x)
        if self.is_eval:
            result = torch.cat([hm, offset], dim=1)
            return result
        else:
            return hm, offset

if __name__ == '__main__':
    net = KeyPointModel(num_classes=4, in_channel=3, is_eval=True)
    input = torch.randn(size=(1, 3, 512, 512))
    output = net(input)
    print(output.shape)
    summary(net, (1, 3, 512, 512))
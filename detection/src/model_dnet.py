"""
    Det-UNet模型定义: 采用UNet作为骨干网络定义的检测模型
"""
import torch
import torch.nn as nn
from torchinfo import summary
from model_unet import UNet

class Backbone(nn.Module):
    """UNet骨干网络"""
    def __init__(self, in_channel, num_classes):
        super(Backbone, self).__init__()
        self.backbone = UNet(in_channel, num_classes, bilinear=True, is_eval=False)
        self.scale = self.backbone.scale

    def forward(self, x):
        x1 = self.backbone.inc(x)
        x2 = self.backbone.down1(x1)
        x3 = self.backbone.down2(x2)
        x4 = self.backbone.down3(x3)
        x5 = self.backbone.down4(x4)
        x = self.backbone.up1(x5, x4)
        reg_feature = self.backbone.up2(x, x3)
        x = self.backbone.up3(reg_feature, x2)
        cls_feature = self.backbone.up4(x, x1)
        return reg_feature, cls_feature

class Neck(nn.Module):
    """特征融合层"""
    def __init__(self, in_channels: list, out_channel: int):
        super(Neck, self).__init__()
        self.layers = nn.ModuleList()
        for channel in in_channels:
            self.layers.append(
                nn.Sequential(nn.Conv2d(channel, out_channel,
                                    kernel_size=1, padding=0, bias=False),
                              nn.BatchNorm2d(out_channel),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(out_channel, out_channel,
                                    kernel_size=3, padding=1, bias=False),
                              nn.BatchNorm2d(out_channel),
                              nn.ReLU(inplace=True)))

    def forward(self, inputs):
        features = [layer(x) for x, layer in zip(inputs, self.layers)]
        return features

class Head(nn.Module):
    """检测头"""
    def __init__(self, num_classes=1, channel=64):
        super(Head, self).__init__()
        # 热力图预测部分
        self.cls_head = nn.Sequential(
            nn.Conv2d(channel, num_classes, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())

        # 宽高预测的部分
        self.wh_head = nn.Sequential(
            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

        # 中心点预测的部分
        self.reg_head = nn.Sequential(
            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

    def forward(self, reg_feature, cls_feature):
        hm = self.cls_head(cls_feature)
        wh = self.wh_head(reg_feature)
        offset = self.reg_head(reg_feature)
        return hm, wh, offset

class DetUNet(nn.Module):
    def __init__(self, in_channel: int = 1, num_classes: int = 1, is_eval: bool = False):
        super(DetUNet, self).__init__()
        self.backbone = Backbone(in_channel, num_classes)
        self.neck = Neck(in_channels=[self.backbone.scale * 2, self.backbone.scale], out_channel=self.backbone.scale)
        self.head = Head(num_classes=num_classes, channel=self.backbone.scale)
        self.is_eval = is_eval
        self.in_channel = in_channel
        self.num_classes = num_classes

    def forward(self, x):
        reg_feature, cls_feature = self.backbone(x)
        reg_feature, cls_feature = self.neck([reg_feature, cls_feature])
        hm, wh, offset = self.head(reg_feature, cls_feature)
        if self.is_eval:
            result = torch.cat([hm, wh, offset], dim=1)
            return result
        else:
            return hm, wh, offset

if __name__ == '__main__':
    net = DetUNet(num_classes=1, in_channel=3, is_eval=True)
    input = torch.randn(size=(1, 3, 512, 512))
    output = net(input)
    summary(net, (1, 3, 512, 512))
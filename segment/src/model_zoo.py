"""Segment model zoo """
from model_module import *

class UNet(nn.Module):
    """Unet model"""
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = False, is_eval: bool = False):
        super(UNet, self).__init__()
        self.is_eval = is_eval
        self.in_channel = n_channels
        self.num_classes = n_classes
        self.bilinear = bilinear
        scale = 4
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
        if self.is_eval == True:
            x = torch.sigmoid(x)
        return x

class PPLiteSeg(nn.Module):
    """PP-LiteSeg model"""
    def __init__(self, num_class=1, n_channel=3, encoder_channels=(32, 64, 256, 512, 1024),
                 encoder_type='stdc1', fusion_type='spatial', act_type='relu'):
        super(PPLiteSeg, self).__init__()
        self.in_channel = n_channel
        self.num_classes = num_class
        decoder_channel_hub = {'stdc1': [32, 64, 128], 'stdc2': [64, 96, 128], 'stdc3': [16, 32, 64]}
        decoder_channels = decoder_channel_hub[encoder_type]

        self.encoder = Encoder(n_channel, encoder_channels, encoder_type, act_type)
        self.sppm = SPPM(encoder_channels[-1], decoder_channels[0], act_type)
        self.decoder = FLD(encoder_channels, decoder_channels, num_class, fusion_type, act_type)

    def forward(self, x):
        size = x.size()[2:]
        x3, x4, x5 = self.encoder(x)
        x5 = self.sppm(x5)
        x = self.decoder(x3, x4, x5, size)

        return x

if __name__ == '__main__':
    """模块测试"""
    from torchinfo import summary
    net = UNet(n_channels=1, n_classes=2, bilinear=True)
    summary(net, (1, 1, 512, 512))
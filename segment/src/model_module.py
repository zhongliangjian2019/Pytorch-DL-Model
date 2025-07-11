"""
@brief 模型网络模块实现
@li 6883/ZhongLiangJian 2023/2/2 13:06
"""
import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    """SE注意力模块"""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        batch, channel, _, _ = input.size()
        output = self.avg_pool(input).view(batch, channel)
        output = self.fc(output).view(batch, channel, 1, 1)
        output = input * output.expand_as(input)
        return output

class CBAM(nn.Module):
    """CBAM注意力模块"""
    def __init__(self, channel1, spatial_kernel=7, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel1, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, input):
        output = input * self.channel_attention(input)
        output = output * self.spatial_attention(output)
        return output

class ChannelAttention(nn.Module):
    """CBAM通道注意力模块"""
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_in =  nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.conv_out = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        avg_out = self.conv_out(self.relu(self.conv_in(self.avg_pool(input))))
        max_out = self.conv_out(self.relu(self.conv_in(self.max_pool(input))))
        output = avg_out + max_out
        output = self.sigmoid(output)
        return output

class SpatialAttention(nn.Module):
    """CBAM空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        avg_out = torch.mean(input, dim=1, keepdim=True)
        max_out, _ = torch.max(input, dim=1, keepdim=True)
        output = torch.cat([avg_out, max_out], dim=1)
        output = self.conv1(output)
        output = self.sigmoid(output)
        return output

class Conv2dBN(Module):
    """带归一化的卷积层"""
    def __init__(self, in_channel, out_channel, kernel=(3, 3), stride=(1, 1), padding=1):
        super(Conv2dBN, self).__init__()
        self.conv2d = nn.Conv2d(in_channel, out_channel, kernel, stride, padding)
        self.batchNorm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        return x

class Conv2dTBN(Module):
    """带归一化的转置卷积层"""
    def __init__(self, in_channel, out_channel, kernel=(3, 3), stride=(2, 2), padding=1, outpadding=1):
        super(Conv2dTBN, self).__init__()
        self.conv2dT = nn.ConvTranspose2d(in_channel, out_channel, kernel, stride, padding, outpadding)
        self.batchNorm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2dT(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        return x

class DownSampleLayer(Module):
    """下采样层"""
    def __init__(self, in_channel: int, out_channel: int,
                 is_pooling: bool = True, is_dropout: bool = False, attention: bool = False):
        super(DownSampleLayer, self).__init__()
        self.layers = nn.ModuleList()
        if is_pooling:
            self.layers.append(nn.MaxPool2d(2, 2))

        self.layers.append(Conv2dBN(in_channel, out_channel))

        if attention:
            self.layers.append(nn.Dropout2d(0.5))
            self.layers.append(Conv2dBN(out_channel, out_channel))
            self.layers.append(nn.Dropout2d(0.5))
        elif is_dropout:
            self.layers.append(CBAM(out_channel, 3, 2))
            self.layers.append(Conv2dBN(out_channel, out_channel))
            self.layers.append(CBAM(out_channel, 3, 2))
        else:
            self.layers.append(Conv2dBN(out_channel, out_channel))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class UpSampleLayer(Module):
    def __init__(self, in_channel: int, out_channel: int, attention: bool = False):
        super(UpSampleLayer, self).__init__()
        self.conv_t = Conv2dTBN(in_channel, out_channel)
        self.attention = CBAM(out_channel * 2, spatial_kernel=3, reduction=2) if attention == True else nn.Dropout2d(0.5)
        self.conv1 = Conv2dBN(out_channel * 2, out_channel)
        self.conv2 = Conv2dBN(out_channel, out_channel)

    def forward(self, x, x1):
        x = self.conv_t(x)
        x = torch.cat([x, x1], dim=1)
        x = self.attention(x)
        x = self.conv2(self.conv1(x))
        return x

def conv3x3(in_channels, out_channels, stride=1):
    """Regular convolution with kernel size 3x3"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    """Regular convolution with kernel size 1x1, a.k.a. point-wise convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

class ConvBNAct(nn.Sequential):
    """Regular convolution -> batchnorm -> activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 bias=False, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):
            padding = (kernel_size - 1) // 2 * dilation
        else:
            raise TypeError("padding value type error")

        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )

class Activation(nn.Module):
    """activation function"""
    def __init__(self, act_type, **kwargs):
        super(Activation, self).__init__()
        activation_hub = {'relu': nn.ReLU, 'relu6': nn.ReLU6,
                          'leakyrelu': nn.LeakyReLU, 'prelu': nn.PReLU,
                          'celu': nn.CELU, 'elu': nn.ELU,
                          'hardswish': nn.Hardswish, 'hardtanh': nn.Hardtanh,
                          'gelu': nn.GELU, 'glu': nn.GLU,
                          'selu': nn.SELU, 'silu': nn.SiLU,
                          'sigmoid': nn.Sigmoid, 'softmax': nn.Softmax,
                          'tanh': nn.Tanh, 'none': nn.Identity,
                          }

        act_type = act_type.lower()
        if act_type not in activation_hub.keys():
            raise NotImplementedError(f'Unsupport activation type: {act_type}')

        self.activation = activation_hub[act_type](**kwargs)

    def forward(self, x):
        return self.activation(x)

class Encoder(nn.Module):
    """PP-LiteSeg Network Encoder Module"""
    def __init__(self, in_channels, encoder_channels, encoder_type, act_type):
        super(Encoder, self).__init__()
        encoder_hub = {'stdc1': STDCBackbone, 'stdc2': STDCBackbone, 'stdc3': STDCBackbone}
        if encoder_type not in encoder_hub.keys():
            raise ValueError(f'Unsupport encoder type: {encoder_type}.\n')

        self.encoder = encoder_hub[encoder_type](in_channels, encoder_channels, encoder_type, act_type)

    def forward(self, x):
        x3, x4, x5 = self.encoder(x)

        return x3, x4, x5

class SPPM(nn.Module):
    """PP-LiteSeg Network SPPM"""
    def __init__(self, in_channels, out_channels, act_type):
        super(SPPM, self).__init__()
        hid_channels = int(in_channels // 4)
        self.act_type = act_type

        self.pool1 = self._make_pool_layer(in_channels, hid_channels, 1)
        self.pool2 = self._make_pool_layer(in_channels, hid_channels, 2)
        self.pool3 = self._make_pool_layer(in_channels, hid_channels, 4)
        self.conv = conv3x3(hid_channels, out_channels)

    def _make_pool_layer(self, in_channels, out_channels, pool_size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            ConvBNAct(in_channels, out_channels, 1, act_type=self.act_type)
        )

    def forward(self, x):
        size = x.size()[2:]
        x1 = F.interpolate(self.pool1(x), size, mode='bilinear', align_corners=True)
        x2 = F.interpolate(self.pool2(x), size, mode='bilinear', align_corners=True)
        x3 = F.interpolate(self.pool3(x), size, mode='bilinear', align_corners=True)
        x = self.conv(x1 + x2 + x3)

        return x

class FLD(nn.Module):
    """PP-LiteSeg Network FLD"""
    def __init__(self, encoder_channels, decoder_channels, num_class, fusion_type, act_type):
        super(FLD, self).__init__()
        self.stage6 = ConvBNAct(decoder_channels[0], decoder_channels[0])
        self.fusion1 = UAFM(encoder_channels[3], decoder_channels[0], fusion_type)
        self.stage7 = ConvBNAct(decoder_channels[0], decoder_channels[1])
        self.fusion2 = UAFM(encoder_channels[2], decoder_channels[1], fusion_type)
        self.stage8 = ConvBNAct(decoder_channels[1], decoder_channels[2])
        self.seg_head = ConvBNAct(decoder_channels[2], num_class, 3, act_type=act_type)

    def forward(self, x3, x4, x5, size):
        x = self.stage6(x5)
        x = self.fusion1(x, x4)
        x = self.stage7(x)
        x = self.fusion2(x, x3)
        x = self.stage8(x)
        x = self.seg_head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x

class STDCBackbone(nn.Module):
    def __init__(self, in_channels, encoder_channels, encoder_type, act_type):
        super(STDCBackbone, self).__init__()
        repeat_times_hub = {'stdc1': [1, 1, 1], 'stdc2': [3, 4, 2], 'stdc3': [1, 1, 1]}
        repeat_times = repeat_times_hub[encoder_type]
        self.stage1 = ConvBNAct(in_channels, encoder_channels[0], 3, 2)
        self.stage2 = ConvBNAct(encoder_channels[0], encoder_channels[1], 3, 2)
        self.stage3 = self._make_stage(encoder_channels[1], encoder_channels[2], repeat_times[0], act_type)
        self.stage4 = self._make_stage(encoder_channels[2], encoder_channels[3], repeat_times[1], act_type)
        self.stage5 = self._make_stage(encoder_channels[3], encoder_channels[4], repeat_times[2], act_type)

    def _make_stage(self, in_channels, out_channels, repeat_times, act_type):
        layers = [STDCModule(in_channels, out_channels, 2, act_type)]

        for _ in range(repeat_times):
            layers.append(STDCModule(out_channels, out_channels, 1, act_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x3 = self.stage3(x)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return x3, x4, x5

class STDCModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act_type):
        super(STDCModule, self).__init__()
        if out_channels % 8 != 0:
            raise ValueError('Output channel should be evenly divided by 8.\n')
        self.stride = stride
        self.block1 = ConvBNAct(in_channels, out_channels // 2, 1)
        self.block2 = ConvBNAct(out_channels // 2, out_channels // 4, 3, stride)
        if self.stride == 2:
            self.pool = nn.AvgPool2d(3, 2, 1)
        self.block3 = ConvBNAct(out_channels // 4, out_channels // 8, 3)
        self.block4 = ConvBNAct(out_channels // 8, out_channels // 8, 3)

    def forward(self, x):
        x = self.block1(x)
        x2 = self.block2(x)
        if self.stride == 2:
            x = self.pool(x)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        return torch.cat([x, x2, x3, x4], dim=1)

class UAFM(nn.Module):
    """PP-LiteSeg Network UAFM"""

    def __init__(self, in_channels, out_channels, fusion_type):
        super(UAFM, self).__init__()
        fusion_hub = {'spatial': SpatialAttentionModule, 'channel': ChannelAttentionModule,
                      'both': [SpatialAttentionModule, ChannelAttentionModule]}
        if fusion_type not in fusion_hub.keys():
            raise ValueError(f'Unsupport fusion type: {fusion_type}.\n')

        self.conv = conv1x1(in_channels, out_channels)
        if fusion_type == 'both':
            self.attention = nn.ModuleList([module(out_channels) for module in fusion_hub[fusion_type]])
        else:
            self.attention = fusion_hub[fusion_type](out_channels)

    def forward(self, x_high, x_low):
        size = x_low.size()[2:]
        x_low = self.conv(x_low)
        x_up = F.interpolate(x_high, size, mode='bilinear', align_corners=True)
        if isinstance(self.attention, nn.ModuleList):
            alpha_0 = self.attention[0](x_up, x_low)
            alpha_1 = self.attention[1](x_up, x_low)
            alpha = alpha_0 * alpha_1
        else:
            alpha = self.attention(x_up, x_low)
        x = alpha * x_up + (1 - alpha) * x_low

        return x

class SpatialAttentionModule(nn.Module):
    """Spatial attention module"""
    def __init__(self, out_channels):
        super(SpatialAttentionModule, self).__init__()
        self.conv = conv1x1(4, 1)

    def forward(self, x_up, x_low):
        mean_up = torch.mean(x_up, dim=1, keepdim=True)
        max_up, _ = torch.max(x_up, dim=1, keepdim=True)
        mean_low = torch.mean(x_low, dim=1, keepdim=True)
        max_low, _ = torch.max(x_low, dim=1, keepdim=True)
        x = self.conv(torch.cat([mean_up, max_up, mean_low, max_low], dim=1))
        x = torch.sigmoid(x)  # [N, 1, H, W]

        return x

class ChannelAttentionModule(nn.Module):
    """Channel attention module"""
    def __init__(self, out_channels):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = conv1x1(4 * out_channels, out_channels)

    def forward(self, x_up, x_low):
        avg_up = self.avg_pool(x_up)
        max_up = self.max_pool(x_up)
        avg_low = self.avg_pool(x_low)
        max_low = self.max_pool(x_low)
        x = self.conv(torch.cat([avg_up, max_up, avg_low, max_low], dim=1))
        x = torch.sigmoid(x)  # [N, C, 1, 1]

        return x

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

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

# Regular convolution with kernel size 3x3
def conv3x3(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                    padding=1, bias=bias)

# Regular convolution with kernel size 1x1, a.k.a. point-wise convolution
def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, 
                    padding=0, bias=bias)

def channel_shuffle(x, groups=2):
    # Codes are borrowed from 
    # https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

# Depth-wise seperable convolution with batchnorm and activation
class DSConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                    dilation=1, act_type='relu', **kwargs):
        super(DSConvBNAct, self).__init__(
            DWConvBNAct(in_channels, in_channels, kernel_size, stride, dilation, act_type, **kwargs),
            PWConvBNAct(in_channels, out_channels, act_type, **kwargs)
        )

# Depth-wise convolution -> batchnorm -> activation
class DWConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                    dilation=1, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):    
            padding = (kernel_size - 1) // 2 * dilation
            
        super(DWConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                        dilation=dilation, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )

# Point-wise convolution -> batchnorm -> activation
class PWConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type='relu', bias=True, **kwargs):
        super(PWConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )

# Regular convolution -> batchnorm -> activation
class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, 
                    bias=False, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):    
            padding = (kernel_size - 1) // 2 * dilation
            
        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )

# Transposed /de- convolution -> batchnorm -> activation
class DeConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=None, 
                    padding=None, act_type='relu', **kwargs):
        super(DeConvBNAct, self).__init__()
        if kernel_size is None:
            kernel_size = 2*scale_factor - 1
        if padding is None:    
            padding = (kernel_size - 1) // 2
        output_padding = scale_factor - 1
        self.up_conv = nn.Sequential(
                                    nn.ConvTranspose2d(in_channels, out_channels, 
                                                        kernel_size=kernel_size, 
                                                        stride=scale_factor, padding=padding, 
                                                        output_padding=output_padding),
                                    nn.BatchNorm2d(out_channels),
                                    Activation(act_type, **kwargs)
                                    )

    def forward(self, x):
        return self.up_conv(x)

class Activation(nn.Module):
    def __init__(self, act_type, **kwargs):
        super(Activation, self).__init__()
        activation_hub = {'relu': nn.ReLU,             'relu6': nn.ReLU6,
                          'leakyrelu': nn.LeakyReLU,    'prelu': nn.PReLU,
                          'celu': nn.CELU,              'elu': nn.ELU, 
                          'hardswish': nn.Hardswish,    'hardtanh': nn.Hardtanh,
                          'gelu': nn.GELU,              'glu': nn.GLU, 
                          'selu': nn.SELU,              'silu': nn.SiLU,
                          'sigmoid': nn.Sigmoid,        'softmax': nn.Softmax, 
                          'tanh': nn.Tanh,              'none': nn.Identity,
                        }
                        
        act_type = act_type.lower()
        if act_type not in activation_hub.keys():
            raise NotImplementedError(f'Unsupport activation type: {act_type}')
        
        self.activation = activation_hub[act_type](**kwargs)
        
    def forward(self, x):
        return self.activation(x)

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, pool_sizes=[1,2,4,6], bias=False):
        super(PyramidPoolingModule, self).__init__()
        assert len(pool_sizes) == 4, 'Length of pool size should be 4.\n'
        hid_channels = int(in_channels // 4)
        self.stage1 = self._make_stage(in_channels, hid_channels, pool_sizes[0])
        self.stage2 = self._make_stage(in_channels, hid_channels, pool_sizes[1])
        self.stage3 = self._make_stage(in_channels, hid_channels, pool_sizes[2])
        self.stage4 = self._make_stage(in_channels, hid_channels, pool_sizes[3])
        self.conv = PWConvBNAct(2*in_channels, out_channels, act_type=act_type, bias=bias)

    def _make_stage(self, in_channels, out_channels, pool_size):
        return nn.Sequential(
                        nn.AdaptiveAvgPool2d(pool_size),
                        conv1x1(in_channels, out_channels)
                )

    def forward(self, x):
        size = x.size()[2:]
        x1 = F.interpolate(self.stage1(x), size, mode='bilinear', align_corners=True)
        x2 = F.interpolate(self.stage2(x), size, mode='bilinear', align_corners=True)
        x3 = F.interpolate(self.stage3(x), size, mode='bilinear', align_corners=True)
        x4 = F.interpolate(self.stage4(x), size, mode='bilinear', align_corners=True)
        x = self.conv(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x


class SegHead(nn.Sequential):
    def __init__(self, in_channels, num_class, act_type, hid_channels=128):
        super(SegHead, self).__init__(
            ConvBNAct(in_channels, hid_channels, 3, act_type=act_type),
            conv1x1(hid_channels, num_class)
        )

"""注意力机制"""
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
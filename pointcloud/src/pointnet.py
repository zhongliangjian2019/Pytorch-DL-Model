"""
@brief PointNet实现
    - PointNetClassifier - 分类模型
    - PointNetSegmenter - 分割模型
@note 实现参考: https://github.com/yanx27/Pointnet_Pointnet2_pytorch#
"""
import torch
from torch import nn

class Conv1D(nn.Module):
    """1D卷积块: conv1d + batch_norm1d + activation"""
    def __init__(self, c_in: int, c_out: int, ksize: int = 1,
                 with_bn: bool = True, act: nn.Module = nn.ReLU()):
        """
        :param c_in: 输入通道数
        :param c_out: 输出通道数
        :param ksize: 卷积核尺寸
        :param with_bn: 是否应用batch_norm
        :param activation: 激活函数
        """
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, ksize)
        self.bn = nn.BatchNorm1d(c_out) if with_bn is True else None
        self.act = act

    def forward(self, x: torch.Tensor):
        """
        :param x: 输入数据, (B, c_in, N)
        :return: 卷积结果, (B, c_out, N)
        """
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x

class Dense(nn.Module):
    """全连接层: linear + batch_norm1d + activation"""
    def __init__(self, c_in: int, c_out: int, with_bn: bool = True, act: nn.Module = nn.ReLU()):
        """
        :param c_in: 输入特征维度
        :param c_out: 输出特征维度
        :param with_bn: 是否应用batch_norm
        :param act: 激活函数
        """
        super().__init__()
        self.fc = nn.Linear(c_in, c_out)
        self.bn = nn.BatchNorm1d(c_out) if with_bn is True else None
        self.act = act

    def forward(self, x: torch.Tensor):
        """
        :param x: 输入数据, (B, c_in)
        :return: 输出结果, (B, c_out)
        """
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x

class TNet(nn.Module):
    """TNet模块"""
    def __init__(self, in_dims: int, out_dims: int, feat_scale: int = 64):
        """
        :param in_dims: 输入特征维度
        :param out_dims: 输出特征维度
        :param feat_scale: 特征提升尺度
        """
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims

        self.conv = nn.Sequential(Conv1D(in_dims, feat_scale),
                                  Conv1D(feat_scale, feat_scale * 2),
                                  Conv1D(feat_scale * 2, feat_scale * 8))

        self.dense = nn.Sequential(Dense(feat_scale * 8, feat_scale * 4),
                                   Dense(feat_scale * 4, feat_scale * 2),
                                   Dense(feat_scale * 2, out_dims * out_dims, act=None))

    def forward(self, x: torch.Tensor):
        """
        :param x: 输入数据, (B, in_dims, N)
        :return: 输出学习的变换矩阵, (B, out_dims, out_dims)
        """
        # 点特征提取
        x = self.conv(x)
        # 提取特征的最大响应
        x = torch.max(x, dim=2)[0] # (B, C)
        # 学习变换矩阵
        x = self.dense(x)
        iden = torch.eye(self.out_dims, device=x.device).view(1, self.out_dims * self.out_dims).repeat(x.size(0), 1)
        x += iden
        x = x.view(-1, self.out_dims, self.out_dims)
        return x

class PointNetEncoder(nn.Module):
    """PointNet编码器模块"""
    def __init__(self, point_dims: int = 3, input_dims: int = 3,
                 glob_feature: bool = True, feat_trans: bool = True, feat_scale: int = 64):
        """
        :param point_dims: 输入数据空间维度2D/3D
        :param input_dims: 输入数据维度(空间维度 + 特征维度)
        :param glob_feature: 输出全局特征(分类) / 全局+局部融合特征(分割)
        :param feat_trans: 是否对特征应用变换矩阵
        :param feat_scale: 特征基准尺度, 用于网络参数规模缩放
        """
        super().__init__()
        # 点空间维度2D/3D
        self.pt_dims = point_dims
        # 输入变换T-Net
        self.input_tnet = TNet(in_dims=input_dims, out_dims=3, feat_scale=feat_scale)
        # 特征变换T-Net
        self.feat_tnet = TNet(in_dims=feat_scale, out_dims=feat_scale, feat_scale=feat_scale) \
            if feat_trans else None
        # 特征提取
        self.mlp1 = nn.Sequential(Conv1D(input_dims, feat_scale),
                                   Conv1D(feat_scale, feat_scale))
        self.mlp2 = nn.Sequential(Conv1D(feat_scale, feat_scale * 2),
                                  Conv1D(feat_scale * 2, feat_scale * 8))
        self.glob_feature = glob_feature
        # 状态信息
        self.in_dims = input_dims
        self.out_dims = feat_scale * 8

    def forward(self, x: torch.Tensor):
        """
        :param x: (B, in_dims, N), in_dims = point_dims + feat_dims
        :return:
        """
        # 输入信息
        B, in_dims, N = x.size()

        # T-Net: 输入变换矩阵学习
        input_trans = self.input_tnet(x)

        # 应用输入变换
        x = x.transpose(2, 1)   # (B, N, in_dims)
        pts, fts = torch.split(x, [self.pt_dims, in_dims - self.pt_dims], dim=-1)
        pts = torch.bmm(pts, input_trans)           # (B, N, pt_dims)
        x = torch.cat([pts, fts], dim=-1)   # (B, N, in_dims)
        x = x.transpose(2, 1)           # (B, in_dims, N)

        # 特征提取: mlp(3,64,64)
        x = self.mlp1(x)   # (B, D1, N)

        # 特征变换学习与应用
        feat_trans = None
        if self.feat_tnet is not None:
            # 特征变换T-Net
            feat_trans = self.feat_tnet(x)  # (B, D1, D1)
            # 应用特征变换
            x = x.transpose(2, 1)
            x = torch.bmm(x, feat_trans)
            x = x.transpose(2, 1)
        # 局部特征
        local_feat = x  # (B, D1, N)

        # 特征强化: mlp(64,128,1024)
        x = self.mlp2(x)

        # 最大池化
        glob_feat = torch.max(x, dim=2)[0]  # (B, D)

        # 输出
        if self.glob_feature:
            return glob_feat, feat_trans    # (B, D), (B, D1, D1)
        else:
            glob_feat = glob_feat.view(-1, glob_feat.size(1), 1).repeat(1, 1, N)    # (B, D, N)
            fuse_feat = torch.cat([glob_feat, local_feat], dim=1)
            return fuse_feat, feat_trans    # (B, D + D1, N), (B, D1, D1)

class PointNetClassifier(nn.Module):
    """PointNet分类器"""
    def __init__(self, point_dims: int = 3, feat_dims: int = 3, class_num: int = 2, feat_scale: int = 64):
        """
        :param point_dims: 点空间维度2D/3D
        :param feat_dims: 点特征维度, rgb/norm
        :param class_num: 分类类别数
        :param feat_scale: 特征基准尺度, 用于网络参数规模缩放
        """
        super().__init__()
        # 特征编码器
        self.feat_encoder = PointNetEncoder(point_dims=point_dims, input_dims=point_dims + feat_dims,
                                            glob_feature=True, feat_trans=True,
                                            feat_scale=feat_scale)
        # 分类头
        encode_scale = self.feat_encoder.out_dims
        self.cla_header = nn.Sequential(Dense(encode_scale, encode_scale // 2),
                                        Dense(encode_scale // 2, encode_scale // 4),
                                        Dense(encode_scale // 4, class_num, with_bn=False,
                                              act=nn.Softmax(dim=1)))

    def forward(self, x: torch.Tensor):
        """
        :param x: (B, D, N)
        :return: (B, class_num)
        """
        # 特征提取
        feat, feat_trans = self.feat_encoder(x)
        # 分类预测
        output = self.cla_header(feat)

        return output, feat_trans

class PointNetSegmenter(nn.Module):
    """PointNet分割器"""
    def __init__(self, point_dims: int = 3, feat_dims: int = 3, class_num: int = 2, feat_scale: int = 64):
        """
        :param point_dims: 点空间维度2D/3D
        :param feat_dims: 点特征维度, rgb/norm
        :param class_num: 分割类别数
        :param feat_scale: 特征基准尺度, 用于网络参数规模缩放
        """
        super().__init__()
        # 编码层
        self.feat_encoder = PointNetEncoder(point_dims=point_dims, input_dims=point_dims + feat_dims,
                                            glob_feature=False, feat_trans=True,
                                            feat_scale=feat_scale)
        # 分割层
        encode_scale = self.feat_encoder.out_dims
        self.seg_header = nn.Sequential(Conv1D(encode_scale + feat_scale, encode_scale // 2),
                                        Conv1D(encode_scale // 2, encode_scale // 4),
                                        Conv1D(encode_scale // 4, encode_scale // 8),
                                        Conv1D(encode_scale // 8, class_num, with_bn=False,
                                               act=nn.Softmax(dim=1)))

    def forward(self, x: torch.Tensor):
        """
        :param x: (B, N, point_dims + feat_dims)
        :return: tuple[(B, N, class_num), (B, D1, D1)]
        """
        # 输入转换
        x = x.transpose(2, 1)   # (B, input_dims, N)
        # 特征编码
        feat, feat_trans = self.feat_encoder(x)
        # 分割预测
        output = self.seg_header(feat)  # (B, class_num, N)
        # 输出转换
        output = output.transpose(2, 1) # (B, N, class_num)

        return output, feat_trans


if __name__ == "__main__":
    """模块测试"""
    # 输入数据
    input = torch.randn([4, 100, 6])
    # 分割模型
    seg_model = PointNetSegmenter(point_dims=3, feat_dims=3, class_num=2)
    output, trans = seg_model(input)
    print("segment: ", output.size(), trans.size())
    # 分类模型
    cla_model = PointNetClassifier(point_dims=3, feat_dims=3, class_num=2)
    output, trans = cla_model(input)
    print("classify: ", output.size(), trans.size())
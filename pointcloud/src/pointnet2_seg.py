import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation


class PointNet2SegMSG(nn.Module):
    """PointNet++多尺度分割模型"""
    def __init__(self, point_dims: int, feat_dims: int, num_classes: int):
        """
        :param num_classes: 分类类别数
        :param point_dims: 点空间维度
        :param feat_dims: 点特征维度
        备注: 分割模型将位置作为特征, 以便捕获结果上下文关联性信息
        """
        super(PointNet2SegMSG, self).__init__()
        self.in_point_dims = point_dims
        self.in_feat_dims = feat_dims
        self.num_classes = num_classes
        self.sa1 = PointNetSetAbstractionMsg(point_dims,1024, [0.05, 0.1], [16, 32],
                                             point_dims+feat_dims, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(point_dims,256, [0.1, 0.2], [16, 32],
                                             32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(point_dims,64, [0.2, 0.4], [16, 32],
                                             128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(point_dims,16, [0.4, 0.8], [16, 32],
                                             256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, xyz: torch.Tensor):
        """
        :param xyz: (B, N, D)， D表示特征维度, N表示点个数
        :return: [B, N, class_num]， 每个点的类别概率
        """
        xyz = xyz.permute(dims=[0, 2, 1])

        l0_points = xyz
        l0_xyz = xyz[:, :self.in_point_dims, :]
        # 下采样层
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        # 上采样层
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = self.softmax(x)
        x = x.permute(0, 2, 1)
        return x


if __name__ == '__main__':
    import torchinfo
    model = PointNet2SegMSG(point_dims=3, feat_dims=3, num_classes=13)
    input = torch.rand(6, 6, 2048)
    output = model(input)
    print(output[0].shape, output[1].shape)
    torchinfo.summary(model, input_size=(6, 6, 2018))
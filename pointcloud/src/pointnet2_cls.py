import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class PointNet2ClsMSG(nn.Module):
    """PointNet++多尺度分类模型"""
    def __init__(self, point_dims: int, feat_dims: int, num_classes: int):
        """
        :param num_classes: 分类类别数
        :param point_dims: 点空间维度
        :param feat_dims: 点特征维度
        备注：分类模型一般不需要xyz信息作为特征，捕获上下文关联信息，更多是捕获全局特征
        """
        super(PointNet2ClsMSG, self).__init__()
        self.in_point_dims = point_dims
        self.in_feat_dims = feat_dims
        self.num_classes = num_classes
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], feat_dims,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz: torch.Tensor):
        B, _, _ = xyz.shape
        if self.in_feat_dims != 0:
            norm = xyz[:, self.in_point_dims:, :]
            xyz = xyz[:, :self.in_point_dims, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x, l3_points


if __name__ == "__main__":
    model = PointNet2ClsMSG(point_dims=3, feat_dims=6, num_classes=13)
    input = torch.randn(size=(6, 9, 2048))
    output = model(input)
    print(output[0].shape, output[1].shape)



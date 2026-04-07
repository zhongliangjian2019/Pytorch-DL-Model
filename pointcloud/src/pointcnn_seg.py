"""
@brief PointCNN分割模型
"""
from pointcnn import PointCNN, RepPointSampling, Dense
from torch import nn
import torch

class PointCNNDown(nn.Module):
    """PointCNN下采样层"""
    def __init__(self, c_in: int, c_out: int, k: int, rep_num: int, dims: int = 3,
                 sampling_method: str = "farthest"):
        """
        :param c_in: 输入点特征维度
        :param c_out: 输出代表点特征维度
        :param k: 邻近点数量
        :param rep_num: 输入代表点数量
        :param dims: 输入点空间维度, 图像-2D/点云-3D
        :param sampling_method: 代表点采样方法, {“random", "farthest"}
        """
        super().__init__()
        self.sampling_func = lambda pts: RepPointSampling(pts, rep_num, sampling_method)
        self.conv = PointCNN(c_in, c_out, dims, k, rep_num)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]):
        """
        :param x: 输入点及其特征
            - pts: 输入点, (B, N, dims)
            - fts: 输入点特征, (B, N, c_in)
        :return: 输出代表点及其特征
            - rep_pts: 代表点, (B, rep_num, dims)
            - rep_fts: 代表点特征, (B, rep_num, c_out)
        """
        pts, fts = x
        rep_pts = self.sampling_func(pts)
        rep_fts = self.conv(rep_pts, pts, fts)
        return rep_pts, rep_fts

class PointCNNUp(nn.Module):
    """PointCNN上采样层"""
    def __init__(self, c_in: int, c_out: int, k: int, rep_num: int, dims: int = 3):
        """
        :param c_in: 输入点特征维度
        :param c_out: 输出代表点特征维度
        :param k: 邻近点数量
        :param rep_num: 输入代表点数量
        :param dims: 输入点空间维度, 图像-2D/点云-3D
        """
        super().__init__()
        self.conv = PointCNN(c_in, c_out, dims, k, rep_num)
        self.dense = Dense(c_in, c_out)

    def forward(self, x_rep: tuple[torch.Tensor, torch.Tensor],
                      x: tuple[torch.Tensor, torch.Tensor]):
        """
        :param x_rep: 代表点及其特征
            - rep_pts: 代表点, (B, rep_num, dims)
            - rep_fts: 代表点特征, (B, rep_num, c_out)
        :param x: 输入点及其特征
            - pts: 输入点, (B, N, dims)
            - fts: 输入点特征, (B, N, c_in)
        :return: 代表点及更新的特征
            - rep_pts: 代表点, (B, rep_num, dims)
            - rep_fts: 代表点更新后的特征, (B, rep_num, c_out)
        """
        rep_pts, rep_fts = x_rep
        pts, fts = x
        rep_fts_up = self.conv(rep_pts, pts, fts)
        # 特征融合: 下采样(浅层) + 上采样(深层)特征
        rep_fts = torch.cat([rep_fts, rep_fts_up], dim=-1)
        rep_fts = self.dense(rep_fts)
        return rep_pts, rep_fts

class PointCNNSegmenter(nn.Module):
    """点云分割器"""
    def __init__(self,
                 point_nums: int = 1024,
                 space_dims: int = 3,
                 in_features: int = 0,
                 num_classes: int = 1,
                 feature_scale: int = 32,
                 ksize: int = 8):
        """
        :param point_nums: 输入数据基准点数, 用于确定下次采样代表点的数量
        :param in_features: 输入点特征维度
        :param feature_scale: 模型特征提取基准尺度
        :param ksize: 邻近点数量
        :param num_classes: 分割类别数
        :param space_dims: 输入点空间维度
        """
        super().__init__()
        self.space_dims = space_dims
        # 输入层
        self.input = PointCNNDown(in_features, feature_scale * 1, ksize, -1, space_dims)
        # 下采样层
        self.down1 = PointCNNDown(feature_scale * 1, feature_scale * 2,  ksize, point_nums // 2,  space_dims)
        self.down2 = PointCNNDown(feature_scale * 2, feature_scale * 4,  ksize, point_nums // 4,  space_dims)
        self.down3 = PointCNNDown(feature_scale * 4, feature_scale * 8,  ksize, point_nums // 8,  space_dims)
        self.down4 = PointCNNDown(feature_scale * 8, feature_scale * 16, ksize, point_nums // 16, space_dims)
        # 上采样层
        self.up1 = PointCNNUp(feature_scale * 16, feature_scale * 8, ksize, point_nums // 8, space_dims)
        self.up2 = PointCNNUp(feature_scale * 8,  feature_scale * 4, ksize, point_nums // 4, space_dims)
        self.up3 = PointCNNUp(feature_scale * 4,  feature_scale * 2, ksize, point_nums // 2, space_dims)
        self.up4 = PointCNNUp(feature_scale * 2,  feature_scale * 1, ksize, -1, space_dims)
        # 输出层
        self.output = nn.Sequential(Dense(feature_scale, feature_scale),
                                    Dense(feature_scale, num_classes, activation=nn.Softmax(dim=-1)))

    def forward(self, input: torch.Tensor):
        """
        :param input: 输入点及其特征
            - pts: 输入点, (B, N, dims)
            - fts: 输入点特征, (B, N, in_features)
        :return: 输出每个点的分类得分 (B, N, num_classes)
        """
        # 切分输入点与特征
        in_pts, in_fts = input[:, :, :self.space_dims], input[:, :, self.space_dims:]
        x = (in_pts, in_fts)
        # 输入
        x_in = self.input(x)

        # 编码阶段
        x_dn1 = self.down1(x_in)
        x_dn2 = self.down2(x_dn1)
        x_dn3 = self.down3(x_dn2)
        x_dn4 = self.down4(x_dn3)

        # 解码阶段
        x_up1 = self.up1(x_dn3, x_dn4)
        x_up2 = self.up2(x_dn2, x_up1)
        x_up3 = self.up3(x_dn1, x_up2)
        x_up4 = self.up4(x_in, x_up3)

        # 输出
        out = self.output(x_up4[1])

        return out

if __name__ == "__main__":
    """模块测试"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PointCNNSegmenter(point_nums=1024, space_dims=3, in_features=6, feature_scale=32, ksize=8, num_classes=4)
    model.to(device=device)
    # 输入
    input = torch.randn([4, 1024, 9], device=device)  # 输入: xyz+rgb+norm[3]
    output = model(input)
    print(output.size())
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor):
    """
    根据索引提取点
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz: torch.Tensor, n_point: int):
    """
    最远点采样
    :param xyz: 点云数据, [B, N, 3]
    :param n_point: 采样点数量, int
    :return: 采样点在点云中的索引, [B, n_point]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, n_point, dtype=torch.long).to(device)    # 采样结果初始化，[B, n_point]
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)   # 最远点索引,(B,)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)        # batch索引,(B,)
    for i in range(n_point):
        # 保存最远点索引
        centroids[:, i] = farthest
        # 根据索引提取最远点
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # 计算最远点与其他点的距离
        dist = torch.sum((xyz - centroid) ** 2, -1)     # [B, N]
        mask = dist < distance
        distance[mask] = dist[mask]     # 更新距离
        farthest = torch.max(distance, -1)[1]   # 提取最大距离索引
    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor):
    """
    球查询搜索临近点
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.n_point = npoint
        self.radius = radius
        self.n_sample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.n_point, self.radius, self.n_sample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetMLPLayer2D(nn.Module):
    """PointNet特征提取MLP层"""
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int = 1):
        super(PointNetMLPLayer2D, self).__init__()
        self.mlp = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size),
                                 nn.BatchNorm2d(out_channel),
                                 nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor):
        """
        :param x: [B, C, H, W]
        :return:
        """
        return self.mlp(x)

class PointNetMLPLayer1D(nn.Module):
    """PointNet特征提取MLP层"""
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int = 1):
        super(PointNetMLPLayer1D, self).__init__()
        self.mlp = nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel_size),
                                 nn.BatchNorm1d(out_channel),
                                 nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor):
        """
        :param x: [B, C, N]
        :return:
        """
        return self.mlp(x)

class PointNetSetAbstractionMsg(nn.Module):
    """集合抽象层: 下采样层"""
    def __init__(self, point_dims: int, n_point: int, radius_list: list, n_sample_list: list, in_channel: int, mlp_list: list):
        """
        :param point_dims:      点的空间维度
        :param n_point:         层采样质心点数量
        :param radius_list:     球查询半径列表
        :param n_sample_list:   球查询最大采样点数量(最大邻域点数量)
        :param in_channel:      输入特征通道数
        :param mlp_list:        PointNet模型尺度列表，list[list]每个元素代表一个尺度的PointNet模型尺度
        """
        super(PointNetSetAbstractionMsg, self).__init__()
        self.point_dims = point_dims
        self.n_point = n_point
        self.radius_list = radius_list
        self.n_sample_list = n_sample_list
        self.mlp_blocks = nn.ModuleList()
        # 多尺度PointNet特征提取层
        for i in range(len(mlp_list)):
            # 第i个尺度的PointNet特征提取层
            mlp_layers = nn.ModuleList()
            last_channel = in_channel + point_dims   # + point_dims 是因为拼接了按质心中心化的点坐标系
            # PointNet的多个堆叠mlp特征提取层
            for out_channel in mlp_list[i]:
                mlp_layers.append(PointNetMLPLayer2D(last_channel, out_channel, 1))
                last_channel = out_channel
            self.mlp_blocks.append(mlp_layers)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # 转换输入形状
        xyz = xyz.permute(0, 2, 1)              # [B, N, C]
        if points is not None:
            points = points.permute(0, 2, 1)    # [B, N, D]
        B, N, C = xyz.shape

        # 采样层：最远点采样质心
        S = self.n_point
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))

        # 分组层+采样层
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            # 分组层：球查询采样邻域点，分组
            # 最大临近点数量
            K = self.n_sample_list[i]
            # 球查询提取临近点
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            # 以质心为中心，平移邻域点
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            # 拼接邻域点和特征
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]

            # 第i个尺度的PointNet特征提取
            for j in range(len(self.mlp_blocks[i])):
                # PointNet层：mlp = conv + bn + relu
                grouped_points = self.mlp_blocks[i][j](grouped_points)
            new_points = torch.max(grouped_points, 2)[0]            # [B, D', S]
            new_points_list.append(new_points)

        # 质心点
        new_xyz = new_xyz.permute(0, 2, 1)
        # 质心点的多尺度特征拼接
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    """点特征传播层: 上采样层"""
    def __init__(self, in_channel: int, mlp_list: list):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_layers = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp_list:
            self.mlp_layers.append(PointNetMLPLayer1D(last_channel, out_channel, 1))
            last_channel = out_channel

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor, points1: torch.Tensor, points2: torch.Tensor):
        """
        将点xyz2对应的特征points2, 按照逆距离加权插值生成点xyz1的传播特征,
        再与点xyz1的原特征points1进行拼接融合, 形成点xyz1的新特征
        :param xyz1: 原始点集, [B, C, N], C表示点的维度, N表示点的个数
        :param xyz2: 与xyz1对应的采样点集, [B, C, S], N > S
        :param points1: xyz1对应的点特征集, [B, D1, N], D1表示点特征维度
        :param points2: xyz2对应的点特征集, [B, D2, S]
        :return: xyz1的新特征, [B, D, N]
        """
        xyz1 = xyz1.permute(0, 2, 1)    # [B, N, C]
        xyz2 = xyz2.permute(0, 2, 1)    # [B, S, C]

        points2 = points2.permute(0, 2, 1)  # [B, S, D]
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # 计算点对点距离
            dists = square_distance(xyz1, xyz2)     # [B, N, S]
            # 排序，取距离最近的3个点
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            # 计算逆距离权重
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            # 逆距离加权得到插值点特征
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        # 插值特征与对应原特征拼接
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        # mlp特征融合与精炼
        new_points = new_points.permute(0, 2, 1)    # [B, D, N]
        for mlp in self.mlp_layers:
            new_points = mlp(new_points)
        return new_points


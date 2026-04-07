"""
@brief PointCNN模型关键模块
"""
import numpy as np
import torch
from torch import nn
from sklearn import neighbors

def KnnIndicesFuncCpu(rep_pts: np.ndarray, pts: np.ndarray, K: int, D: int):
    """
    K近邻搜索 - CPU版本
    :param rep_pts: 代表点集 (B, P, dims)
    :param pts: 源点集
    :param K: 邻近点数量
    :param D: 扩张率
    :return: 每个代表点的邻近点索引
    """
    region_idx = []
    for i in range(rep_pts.shape[0]):
        # 样本点集
        samples = pts[i]
        # 查询点集
        queries = rep_pts[i]
        # 模型训练
        neigh = neighbors.NearestNeighbors(n_neighbors=K * D + 1, algorithm="ball_tree")
        neigh.fit(samples)
        # 检索查询点的最近邻
        indices = neigh.kneighbors(queries, return_distance=False)
        region_idx.append(indices[:, 1::D])
    region_idx = torch.from_numpy(np.stack(region_idx, axis=0))
    return region_idx

class Dense(nn.Module):
    """全连接层: 线性层 + 激活函数 + Dropout层"""
    def __init__(self, c_in: int, c_out: int, drop_rate: float = 0, activation: callable(torch.Tensor) = nn.ReLU()):
        """
        :param c_in: 输入特征维度
        :param c_out: 输出特征维度
        """
        super().__init__()
        self.fc = nn.Linear(c_in, c_out)
        self.activation = activation
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else None

    def forward(self, x):
        x = self.fc(x)
        if self.activation:
            x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class Conv2DBA(nn.Module):
    """2D卷积块: Conv2d + BN + Activation"""
    def __init__(self, c_in: int, c_out: int, ksize: tuple[int, int], with_bn: bool = True,
                 activation: callable(torch.Tensor) = nn.ReLU()):
        """
        :param c_in: 输入通道数
        :param c_out: 输出通道数
        :param ksize: 卷积核尺寸
        :param with_bn: 是否应用batch_normal
        :param activation: 激活函数
        """
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, ksize, bias=not with_bn)
        self.activation = activation
        self.bn = nn.BatchNorm2d(c_out, momentum=0.9) if with_bn else None

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x

class DepthSepConv(nn.Module):
    """深度可分离卷积"""
    def __init__(self, c_in: int, c_out: int, ksize: tuple[int, int], depth_multiplier: int = 1,
                 with_bn: bool = True, activation: callable(torch.Tensor) = nn.ReLU()):
        """
        :param c_in: 输入通道数
        :param c_out: 输出通道数
        :param ksize: 卷积核尺寸
        :param depth_multiplier: 深度卷积的深度比率
        :param with_bn: 是否应用batch_normal
        :param activation: 激活函数
        """
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(c_in, c_in * depth_multiplier, ksize, groups=c_in),
                                  nn.Conv2d(c_in * depth_multiplier, c_out, 1, bias=not with_bn))
        self.activation = activation
        self.bn = nn.BatchNorm2d(c_out, momentum=0.9) if with_bn else None

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x

def EndChannels(f):
    """类装饰器: 以对末尾通道作为特征应用2D卷积"""
    class WrappedLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = f

        def forward(self, x):
            x = x.permute(0, 3, 1, 2)
            x = self.f(x)
            x = x.permute(0, 2, 3, 1)
            return x
    return WrappedLayer()

class XConv(nn.Module):
    """X-Conv模块"""
    def __init__(self, c_in: int, c_out: int, dims: int, c_mid: int, k: int, rep_num: int, depth_multiplier: int):
        """
        :param c_in: 输入点特征维度
        :param c_out: 输出代表点特征维度
        :param dims: 输入点空间维度, 图像-2D/点云-3D
        :param c_mid: 输入点坐标特征提升维度
        :param k: 邻近点数量
        :param rep_num: 输入代表点数量
        :param depth_multiplier: 深度可分离卷积, 深度卷积过程提升倍数
        """
        super().__init__()
        self.C_in = c_in
        self.C_out = c_out
        self.dims = dims
        self.K = k
        self.P = rep_num
        # 点坐标特征提升层
        self.pts_lifted_mlp = nn.Sequential(Dense(dims, c_mid),
                                            Dense(c_mid, c_mid))
        # X变换层
        self.x_transform = nn.Sequential(EndChannels(Conv2DBA(c_in=dims, c_out=k*k, ksize=(1, k), with_bn=False)),
                                         Dense(c_in=k*k, c_out=k*k),
                                         Dense(c_in=k*k, c_out=k*k, activation=None))
        # 联合特征学习层
        self.rep_fts_mlp = EndChannels(DepthSepConv(c_in=c_mid + c_in, c_out=c_out, ksize=(1, k),
                                                    depth_multiplier=depth_multiplier))

    def forward(self, rep_pts: torch.Tensor, pts: torch.Tensor, fts: torch.Tensor):
        """
        对输入数据应用X-Conv提取代表点特征
        :param rep_pts: 代表点集           (B, P, dims)
        :param pts:     代表点邻域点集      (B, P, K, dims)
        :param fts:     代表点领域点集特征   (B, P, K, C_in)
        :return:        聚合到代表点的特征   (B, P, C_out)
        """
        # 输入形状检查
        if fts is not None:
            assert(rep_pts.size(0) == pts.size(0) == fts.size(0))
            assert(rep_pts.size(1) == pts.size(1) == fts.size(1))
            assert(pts.size(2) == fts.size(2) == self.K)
            assert(fts.size(3) == self.C_in)
        else:
            assert(rep_pts.size(0) == pts.size(0))
            assert(rep_pts.size(1) == pts.size(1))
            assert(pts.size(2) == self.K)
        assert(rep_pts.size(2) == pts.size(3) == self.dims)

        # step1: P' ← P - p, 转换代表点的邻近点到局部坐标系
        p_center = torch.unsqueeze(rep_pts, dim=2)  # (B, P, dims) -> (B, P, 1, dims)
        pts_local = pts - p_center                  # (B, P, K, dims)

        # step2: Fδ ← MLPδ(P'), 点坐标特征提升
        pts_lifted_fts = self.pts_lifted_mlp(pts_local) # (B, P, K, C_mid)

        # step3: F* ← [Fδ, F], 特征拼接
        if fts is None:
            fts_cat = pts_lifted_fts
        else:
            fts_cat = torch.cat([pts_lifted_fts, fts], dim=-1)  # (B, P, K, C_mid + C_in)

        # step4: X ← MLP(P'), 学习X变换矩阵
        X = self.x_transform(pts_local)
        B = fts_cat.size(0)
        X = X.view(B, self.P, self.K, self.K)

        # step5: Fx ← X x F*, 应用X变换
        fts_x = torch.matmul(X, fts_cat)

        # step6: Fp ← Conv(K, Fx), 学习代表点特征
        rep_fts = self.rep_fts_mlp(fts_x)
        rep_fts = rep_fts.squeeze(dim=2)
        return rep_fts

class PointCNN(nn.Module):
    """点卷积模块"""
    def __init__(self, c_in: int, c_out: int, dims: int, k: int,
                 rep_num: int, dilate: int = 1, nn_indices_func: callable = KnnIndicesFuncCpu):
        """
        :param c_in: 输入点特征维度
        :param c_out: 输出代表点特征维度
        :param dims: 输入点空间维度, 图像-2D/点云-3D
        :param k: 邻近点数量
        :param rep_num: 输入代表点数量
        :param dilate: 扩展率
        :param nn_indices_func: 邻近点搜索函数
        """
        super().__init__()
        # 计算点坐标提升维度
        c_mid = c_out // 2 if c_in == 0 else c_out // 4
        # 计算深度卷积倍率
        if c_in == 0:
            depth_multiplier = 1
        else:
            depth_multiplier = min(int(np.ceil(c_out / c_in)), 4)
        # 初始化邻近点搜索函数
        self.nn_indices_func = lambda rep_pts, pts: nn_indices_func(rep_pts, pts, k, dilate)
        # 特征强化层
        self.dense = Dense(c_in, c_out // 2) if c_in != 0 else None
        # X-Conv层
        self.x_conv = XConv(c_out // 2 if c_in != 0 else c_in, c_out, dims, c_mid, k, rep_num, depth_multiplier)

    def select_region(self, pts: torch.Tensor, pts_idx: torch.Tensor):
        """
        根据索引选择数据
        :param pts: 数据
        :param pts_idx: 数据索引
        :return:
        """
        regions = torch.stack([pts[n][idx, :] for n, idx in enumerate(torch.unbind(pts_idx, dim=0))], dim=0)
        return regions

    def forward(self, rep_pts: torch.Tensor, pts: torch.Tensor, fts: torch.Tensor = None):
        # 特征强化
        fts = self.dense(fts) if fts is not None else None
        # 获取邻近点索引
        pts_idx = self.nn_indices_func(rep_pts.cpu(), pts.cpu())
        pts_idx.to(device=rep_pts.device)
        # 获取邻近点及其特征
        pts_regional = self.select_region(pts, pts_idx)
        fts_regional = self.select_region(fts, pts_idx) if fts is not None else fts
        # x-conv卷积
        rep_fts = self.x_conv(rep_pts, pts_regional, fts_regional)
        return rep_fts

def RepPointSampling(pts: torch.Tensor, sample_num: int, method: str = "random"):
    """
    代表点采样
    :param pts: 输入点集, (B, N, dims)
    :param sample_num: 采样数
    :param method: 采样方法, "random" - 随机采样, "farthest" - 最远点采样
    :return: 采样的代表点, (B, sample_num, dims)
    """

    def farthest_point_sampling_batch(points, k):
        """
        Farthest Point Sampling (FPS) algorithm for batched point clouds.

        Parameters:
        points (numpy.ndarray): Input point cloud with shape (B, N, D),
                                where B is the batch size, N is the number of points, and D is the dimension.
        k (int): Number of points to sample.

        Returns:
        numpy.ndarray: Sampled point indices with shape (B, k).
        """
        B, N, D = points.shape
        if k >= N:
            return np.arange(N, dtype=int).reshape(1, -1).repeat(B, axis=0)

        # Initialize the array to store the sampled indices
        sampled_indices = np.zeros((B, k), dtype=int)

        # Randomly select the first point for each batch
        first_indices = np.random.randint(0, N, size=B)
        sampled_indices[:, 0] = first_indices

        # Initialize the minimum distances to a large number for each batch
        min_distances = np.full((B, N), np.inf)

        for b in range(B):
            # Update the minimum distances for the first point
            first_point = points[b, first_indices[b]]
            distances = np.linalg.norm(points[b] - first_point, axis=1)
            min_distances[b] = distances

        for i in range(1, k):
            # Select the point with the maximum distance for each batch
            farthest_indices = np.argmax(min_distances, axis=1)
            sampled_indices[:, i] = farthest_indices

            # Update the minimum distances for the newly sampled point
            for b in range(B):
                farthest_point = points[b, farthest_indices[b]]
                distances = np.linalg.norm(points[b] - farthest_point, axis=1)
                min_distances[b] = np.minimum(min_distances[b], distances)

        return sampled_indices

    rep_pts = pts
    if 0 < sample_num < pts.size(1):
        if method == "random":
            """随机采样 - 分类"""
            idx = np.random.choice(pts.size()[1], sample_num, replace=False).tolist()
            rep_pts = pts[:, idx, :]
        else:
            """最远点采样 - 分割"""
            sample_idx = farthest_point_sampling_batch(pts.cpu().numpy(), sample_num)
            batch_idx = np.arange(pts.size()[0])[:, None].repeat(sample_num, axis=1)
            rep_pts = pts[batch_idx, sample_idx]
    return rep_pts

if __name__ == "__main__":
    """单元测试"""
    # 模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PointCNN(c_in=6, c_out=8, dims=3, k=8, rep_num=32, dilate=1, nn_indices_func=KnnIndicesFuncCpu)
    model.to(device=device)
    # 输入
    rep_pts = torch.randn(size=(4, 32, 3), device=device)
    pts = torch.randn(size=(4, 256, 3), device=device)
    fts = torch.randn(size=(4, 256, 6), device=device)
    # 输出
    rep_fts = model(rep_pts, pts, fts)
    print(rep_fts.size())

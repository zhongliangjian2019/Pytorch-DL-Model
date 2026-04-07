"""
@brief 点云数据增强
"""
import numpy as np

class DataAugment:
    """数据增强"""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, points: np.ndarray, labels: np.ndarray):
        """数据增强"""
        # 随机丢弃点
        if np.random.random() > self.p:
            points, labels = random_point_dropout(points, labels, max_dropout_ratio=0.85)
        # 随机旋转点云
        if np.random.random() > self.p:
            points[:, :3] = rotate_point_cloud_z(points[:, :3])
        # 随机旋转点云
        if np.random.random() > self.p:
            points[:, :3] = rotate_point_cloud(points[:, :3])
        # 随机打乱点云点
        if np.random.random() > self.p:
            points, labels = shuffle_points(points, labels)
        # 随机扰动点云
        if np.random.random() > self.p:
            points[:, :3] = rotate_perturbation_point_cloud(points[:, :3])
        # 随机噪声
        if np.random.random() > self.p:
            points[:, :3] = add_noise_point_cloud(points[:, :3])
        # 随机缩放
        if np.random.random() > self.p:
            points[:, :3] = random_scale_point_cloud(points[:, :3])
        # 随机平移
        if np.random.random() > self.p:
            points[:, :3] = random_shift_point_cloud(points[:, :3], shift_range=0.1)

        return points, labels

def normalize_data(data: np.ndarray):
    """ 归一化数据：根据块中心进行归一化
    :param data: 输入点云, Nx3
    :return 归一化数据
    """
    centroid = np.mean(data, axis=0)
    normal_data = data - centroid
    m = np.max(np.sqrt(np.sum(normal_data ** 2, axis=1)))
    normal_data /= m
    return normal_data

def shuffle_points(data: np.ndarray, label: np.ndarray):
    """ 打乱点云中点的顺序 -- 改变最远点采样的行为.
        Input:
            data: NxC
            label: N,1
        Output:
            data, label
    """
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    return data[idx,:], label[idx]

def rotate_point_cloud(data: np.ndarray):
    """
    沿上方向随机旋转点云0-180度
    :param data: 输入点云, Nx3
    :return rotated_data: 旋转后的点云, Nx3
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(data, rotation_matrix)
    return rotated_data

def rotate_point_cloud_z(points: np.ndarray):
    """
    绕Z轴随机旋转0-180度点云
    :param points: 输入数据, N x 3
    :return: 输出旋转后的数据, N x 3
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cos_val = np.cos(rotation_angle)
    sin_val = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_val, sin_val, 0],
                                [-sin_val, cos_val, 0],
                                [0, 0, 1]])
    rotated_data = np.dot(points, rotation_matrix)
    return rotated_data

def rotate_point_cloud_with_normal(data: np.ndarray):
    ''' 随机旋转点云和法向量
        Input:
            Nx6, 数据为xyz[3] + normal[3]
        Output:
            Nx6
    '''
    rotated_data = data.copy()
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data[:,0:3] = np.dot(data[:, 0:3], rotation_matrix)
    rotated_data[:,3:6] = np.dot(data[:, 3:6], rotation_matrix)
    return rotated_data

def rotate_perturbation_point_cloud_with_normal(data, angle_sigma=0.06, angle_clip=0.18):
    """ 通过小的旋转进行点云扰动
        Input:
          Nx6 array, original batch of point clouds and point normals
        Return:
          Nx3 array, rotated batch of point clouds
    """
    rotated_data = data.copy()
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    rotated_data[:,0:3] = np.dot(data[:, 0:3], R)
    rotated_data[:,3:6] = np.dot(data[:, 3:6], R)
    return rotated_data

def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k,:,0:3]
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx6 array, original batch of point clouds with normal
          scalar, angle of rotation
        Return:
          BxNx6 array, rotated batch of point clouds iwth normal
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k,:,0:3]
        shape_normal = batch_data[k,:,3:6]
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k,:,3:6] = np.dot(shape_normal.reshape((-1,3)), rotation_matrix)
    return rotated_data

def rotate_perturbation_point_cloud(data: np.ndarray, angle_sigma: float = 0.06, angle_clip: float = 0.18):
    """
    随机抖动点云
    :param data: 输入点云, Nx3
    :return rotated_data: 输出点云, Nx3
    """
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    rotated_data = np.dot(data, R)
    return rotated_data

def add_noise_point_cloud(data: np.ndarray, sigma: float = 0.01, clip: float = 0.05):
    """
    对点云添加高斯噪声
    :param data: 点云数据, Nx3
    :param sigma: 噪声方差
    :param clip: 裁剪尺度
    :returns noised_data: 添加噪声的点云, Nx3
    """
    N, C = data.shape
    assert(clip > 0)
    noise = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    noised_data = data + noise
    return noised_data

def random_shift_point_cloud(data: np.ndarray, shift_range: float = 0.1):
    """
    随机平移点云
    :param data: 输入点云, Nx3
    :param shift_range: 平移尺度范围
    :returns shift_data: 平移后的点云, Nx3
    """
    shifts = np.random.uniform(-shift_range, shift_range, (1, 3))
    shift_data = data + shifts
    return shift_data

def random_scale_point_cloud(data: np.ndarray, scale_low: float = 0.8, scale_high: float = 1.25):
    """
    随机缩放点云
    :param data: 输入点云, Nx3
    :param scale_low: 缩放尺度低值
    :param scale_high: 缩放尺度高值
    :return: scale_data: 缩放后的点云, Nx3
    """
    scale = np.random.uniform(scale_low, scale_high)
    scale_data = data * scale
    return scale_data

def random_point_dropout(points: np.ndarray,
                         labels: np.ndarray = None,
                         max_dropout_ratio: float = 0.875):
    """
    随机丢弃点云数据点
    :param batch_points: 数据点, NxC
    :param batch_labels: 标签, N
    :param max_dropout_ratio: 最大丢弃比例
    :return: 处理结果
    """
    dropout_ratio =  np.random.random() * max_dropout_ratio # 0~0.875
    drop_idx = np.where(np.random.random((points.shape[1])) <= dropout_ratio)[0]
    if len(drop_idx) > 0:
        points[drop_idx, :] = points[0, :] # set to the first point
        if labels is not None and len(labels) > 1:
            labels[drop_idx] = labels[0]
    return points, labels

if __name__ == "__main__":
    points = np.random.random(size=(100, 9))
    labels = np.random.randint(0, 5, size=(100, 1))
    augment = DataAugment(0.5)
    points, labels = augment(points, labels)
    print(points.shape, labels.shape)


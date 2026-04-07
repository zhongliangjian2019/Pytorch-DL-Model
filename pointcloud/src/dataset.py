"""
@brief 点云数据集划分与加载
"""
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from tqdm import tqdm
from data_augment import DataAugment, normalize_data
import torch
import logging

class PCDDataset(Dataset):
    """
    点云数据集
    目录结构:
    - data_dir
        - datas: 数据目录
            - pcd1.txt
            - pcd2.txt
            - ...
        - train.txt: 训练集文件名, 如pcd1.txt, pcd2.txt
        - val.txt
        - test.txt
    数据格式:
    - 每个包含数据点信息和标签信息, 每行为1个数据点, 如 x,y,z,r,g,b,label
    """

    def __init__(self, data_root: str , num_classes: int = 13, mode: str = 'train', sample_point_num: int = 4096,
                 sample_block_size: float = 1.0, sample_rate: float = 1.0, transform: bool = True, use_ratio: float = 1.0):
        """
        :param data_root: 数据根目录
        :param num_classes: 数据分类类别数
        :param mode: 数据加载模式, {'train', 'val'}
        :param sample_point_num: 采样数据点数
        :param sample_block_size: 采样块大小
        :param sample_rate: 采样率
        :param transform: 是否应用增强变换
        :param use_ratio: 数据集使用比例, 该参数主要用于小批数据测试
        """
        super().__init__()
        self.logger = logging.getLogger("dataset")

        self.data_dir = os.path.join(data_root, "datas")
        self.num_point = sample_point_num
        self.block_size = sample_block_size
        self.transform = DataAugment(0.5) if transform is True else None

        self.data_files = None
        self.class_weights = None
        self.class_labels = None

        self.data_buffer = {}

        # 检查数据信息文件
        # 样本文件 / 标签文件
        sample_file = os.path.join(data_root, mode + "_samples.txt")
        label_file = os.path.join(data_root, mode + "_label_infos.txt")
        if not os.path.exists(sample_file) or not os.path.exists(label_file):
            self.logger.info(f"Not found {os.path.basename(sample_file)} and {os.path.basename(label_file)}, create file ...")
            calculate_dataset_infos(data_root, num_classes, mode, sample_point_num, sample_rate)
            self.logger.info("create file finished.")

        self.logger.info(f"Found {os.path.basename(sample_file)} and {os.path.basename(label_file)} files.")
        # 加载样本
        self.logger.info("Load sample info: ")
        with open(sample_file, 'r') as file:
            self.data_files = [name.strip('\n') for name in file.readlines()]
        self.data_files = self.data_files[:int(len(self.data_files) * use_ratio)]
        self.logger.info("Total {} samples in {} set.".format(len(self.data_files), mode))

        # 加载标签信息
        self.logger.info("Load label info: ")
        with open(label_file, 'r') as file:
            self.class_weights = []
            self.class_labels = {}
            for line in file.readlines():
                info = line.strip('\n').split(' ')
                if len(info) != 3:
                    continue
                class_id, class_name, class_weight = info
                self.class_labels[int(class_id)] = class_name
                self.class_weights.append(float(class_weight))
        assert len(self.class_weights) == num_classes, (
                "error: num_classes(%d) not match label_infos(%d)." % (num_classes, len(self.class_weights)))
        self.logger.info(f"Load finished, num_classes = {len(self.class_weights)}")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx: int):
        """
        获取一个训练样本(点云块及其标签)
        参数:
            idx (int): 样本索引
        返回:
            current_points (np.array): 处理后的点云数据，形状为(num_point, 9)
            current_labels (np.array): 对应的点云标签，形状为(num_point,)
        """
        # 根据索引确定从哪个房间采样
        filename = self.data_files[idx]
        if filename in self.data_buffer.keys():
            data = self.data_buffer[filename]
        else:
            data = np.loadtxt(os.path.join(self.data_dir, filename), delimiter=' ', skiprows=1)
            self.data_buffer[filename] = data
        points, labels = data[:, :-1], data[:, -1]
        point_num = points.shape[0]
        # 随机中心点采样
        while True:
            # 随机选择一个点作为中心(只取xyz)
            center = points[np.random.choice(point_num), :3]
            # 计算块的边界
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            # 找到在当前块内的所有点的索引
            point_ids = np.where(
                (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) &
                (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1])
            )[0]
            # 如果块内点数大于1024，则接受这个块
            if point_ids.size > 1024:
                break

        # 从块中采样固定数量的点
        if point_ids.size >= self.num_point:
            # 点数足够时，无放回抽样
            select_point_ids = np.random.choice(point_ids, self.num_point, replace=False)
        else:
            # 点数不足时，有放回抽样(重复采样)
            select_point_ids = np.random.choice(point_ids, self.num_point, replace=True)

        # 归一化处理点云数据
        selected_points = points[select_point_ids, :]   # 获取选中的点，num_point * 6
        current_points = np.zeros((self.num_point, points.shape[1]))  # 初始化输出数组，num_point * 9

        # 添加全局归一化坐标(相对于整个房间)
        # points_coord_max = np.max(points[:, :3], axis=0)
        # points_coord_min = np.min(points[:, :3], axis=0)
        # current_points[:, 6] = selected_points[:, 0] / points_coord_max[0]
        # current_points[:, 7] = selected_points[:, 1] / points_coord_max[1]
        # current_points[:, 8] = selected_points[:, 2] / points_coord_max[2]

        # 坐标中心化(相对于块中心)
        selected_points[:, :3] = normalize_data(selected_points[:, :3])

        # 颜色归一化(0-255 -> 0-1)
        selected_points[:, 3:6] /= 255.0

        # 将处理后的点云数据填充到输出数组的前6个通道
        current_points[:, 0:6] = selected_points

        # 获取对应的标签
        current_labels = labels[select_point_ids]

        # 应用数据增强变换(如果提供)
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)

        return (torch.as_tensor(current_points, dtype=torch.float32),
                torch.as_tensor(current_labels, dtype=torch.int64))

def dataset_split(data_dir: str, train_ratio: float = 0.8):
    """划分数据集"""
    filenames = [name for name in os.listdir(data_dir) if name.endswith('txt')]
    ids = np.random.choice(len(filenames), size=int(len(filenames) * train_ratio), replace=False).tolist()
    train_files = [filenames[id] + "\n" for id in ids]
    val_files = [filenames[id] + "\n" for id in range(len(filenames)) if id not in ids]

    with open(os.path.join(os.path.dirname(data_dir), "train.txt"), 'w') as file:
        file.writelines(train_files)

    with open(os.path.join(os.path.dirname(data_dir), "val.txt"), 'w') as file:
        file.writelines(val_files)

def calculate_dataset_infos(data_dir: str , num_classes: int = 14, mode: str = 'train',
                            sample_point_num: int = 4096, sample_rate: float = 1.0):
        """
        计算数据集信息, 包括类别权重, 样本采样频率
        :param data_dir: 数据根目录
        :param num_classes: 数据分类类别数
        :param mode: 数据加载模式, {'train', 'val'}
        :param sample_point_num: 采样数据点数
        :param sample_rate: 采样率
        """
        # 加载训练文件列表
        txt_file = os.path.join(data_dir, mode + ".txt")
        with open(txt_file, 'r') as file:
            data_files = [name.strip('\n') for name in file.readlines()]

        # 初始化数据信息列表
        data_point_nums = []                        # 存储点云点数量
        label_weights = np.zeros(num_classes)       # 初始化类别的权重统计

        # 加载数据并统计信息
        with tqdm(total=len(data_files), desc="load data") as pbar:
            for name in tqdm(data_files, total=len(data_files)):
                # 点云格式为: (point, label), point: (x,y,z,r,g,b,...)
                data = np.loadtxt(os.path.join(data_dir, "datas", name), skiprows=1)
                # 获取数据标签
                labels = data[:, -1]
                # 统计类别频率
                freq, _ = np.histogram(labels, np.arange(num_classes + 1))
                label_weights += freq
                # 存储数据信息
                data_point_nums.append(len(labels))
                pbar.update()

        # 计算类别权重: 使用逆向频率加权，处理类别不平衡问题
        label_weights = label_weights.astype(np.float32)
        label_weights = label_weights / np.sum(label_weights)  # 归一化为概率分布
        label_weights = np.power(np.max(label_weights) / (label_weights + 1e-8), 1 / 3.0) # 使用立方根平滑权重，减少极端不平衡的影响
        print("Class weights:", label_weights)

        # 根据点云尺度计算每个数据文件采样次数
        # 计算总采样次数: 总点数 * 采样率 / 每个块的点数
        sample_all_num = int(np.sum(data_point_nums) * sample_rate / sample_point_num)
        # 根据采样概率为每个房间生成对应的采样次数
        # 计算采样概率
        sample_probs = np.asarray(data_point_nums) / np.sum(data_point_nums)
        data_ids = []
        for id, name in enumerate(data_files):
            prob = sample_probs[id]
            sample_num = int(prob * sample_all_num + 1)
            data_ids.extend([id] * sample_num)
        print("Total {} samples in {} set.".format(len(data_ids), mode))

        # 保存标签信息到文件
        label_infos_path = os.path.join(data_dir, mode + "_label_infos.txt")
        class_labels = {'background': 0, 'chicken': 1}
        label_infos = []
        for label, id in class_labels.items():
            weight = label_weights[id]
            info = "%d %s %.6f\n" % (id, label, weight)
            label_infos.append(info)
        with open(label_infos_path, 'w') as file:
            file.writelines(label_infos)

        # 保存样本采样信息到文件
        sample_infos = []
        for id in data_ids:
            sample_infos.append(data_files[id] + "\n")
        sample_infos_path = os.path.join(data_dir, mode + "_samples.txt")
        with open(sample_infos_path, 'w') as file:
            file.writelines(sample_infos)


if __name__ == "__main__":
    # calculate_dataset_infos(data_dir="../datasets/chicken3d", num_classes=2, mode='val', sample_point_num=2048)
    dataset = PCDDataset(data_root="../datasets/chicken3d", num_classes=2)
    for i in range(10):
        points, labels = dataset[i]
        print(points.shape, labels.shape)
    # dataset_split(data_dir="..\datasets\chicken3d\data", train_ratio=0.8)


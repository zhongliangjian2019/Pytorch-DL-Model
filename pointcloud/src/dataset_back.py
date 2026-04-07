"""
@brief 点云数据集加载
"""
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import open3d as o3d
from tqdm import tqdm
from data_augment import DataAugment

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

    def __init__(self, data_dir: str , num_classes: int = 13, mode: str = 'train', sample_point_num: int = 4096,
                 sample_block_size: float = 1.0, sample_rate: float = 1.0, transform: bool = True):
        """
        :param data_dir: 数据根目录
        :param num_classes: 数据分类类别数
        :param mode: 数据加载模式, {'train', 'val'}
        :param sample_point_num: 采样数据点数
        :param sample_block_size: 采样块大小
        :param sample_rate: 采样率
        :param transform: 是否应用增强变换
        """
        super().__init__()
        self.num_point = sample_point_num
        self.block_size = sample_block_size
        self.transform = DataAugment(0.5) if transform is True else None

        # 加载训练文件列表
        txt_file = os.path.join(data_dir, mode + ".txt")
        with open(txt_file, 'r') as file:
            data_files = [name.strip('\n') for name in file.readlines()]

        # 初始化数据信息列表
        self.data_points = []               # 存储点云数据(x,y,z,r,g,b)
        self.data_labels = []               # 存储点云标签
        self.data_coord_min = []            # 存储点云最小坐标
        self.data_coord_max = []            # 存储点云最大坐标
        data_point_nums = []                # 存储点云点数量
        label_weights = np.zeros(num_classes)        # 初始化类别的权重统计

        # 加载数据并统计信息
        with tqdm(total=len(data_files), desc="load data") as pbar:
            for name in tqdm(data_files, total=len(data_files)):
                # 点云格式为: (point, label), point: (x,y,z,r,g,b,...)
                data = np.loadtxt(os.path.join(data_dir, "datas", name))
                # 划分数据点与标签
                points, labels = data[:, :-1], data[:, -1]
                # 统计类别频率
                freq, _ = np.histogram(labels, np.arange(num_classes + 1))
                label_weights += freq
                # 计算数据坐标范围
                coord_min, coord_max = np.min(points[:, :3], axis=0), np.max(points[:, :3], axis=0)
                # 存储数据信息
                self.data_points.append(points)
                self.data_labels.append(labels)
                self.data_coord_min.append(coord_min)
                self.data_coord_max.append(coord_max)
                data_point_nums.append(len(labels))
                pbar.update()

        # 计算类别权重: 使用逆向频率加权，处理类别不平衡问题
        label_weights = label_weights.astype(np.float32)
        label_weights = label_weights / np.sum(label_weights)  # 归一化为概率分布
        self.label_weights = np.power(np.max(label_weights) / label_weights, 1 / 3.0) # 使用立方根平滑权重，减少极端不平衡的影响
        print("Class weights:", self.label_weights)

        # 根据点云尺度计算每个数据文件采样次数
        # 计算总采样次数: 总点数 * 采样率 / 每个块的点数
        sample_all_num = int(np.sum(data_point_nums) * sample_rate / sample_point_num)
        # 根据采样概率为每个房间生成对应的采样次数
        # 计算采样概率
        sample_probs = np.asarray(data_point_nums) / np.sum(data_point_nums)
        self.data_ids = []
        for id in range(len(data_files)):
            prob = sample_probs[id]
            sample_num = int(prob * sample_all_num + 1)
            self.data_ids.extend([id] * sample_num)
        print("Total {} samples in {} set.".format(len(self.data_ids), mode))

    def __len__(self):
        return len(self.data_ids)

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
        data_id = self.data_ids[idx]
        points = self.data_points[data_id]
        labels = self.data_labels[data_id]
        point_num = points.shape[0]  # 当前房间的总点数
        # 随机采样一个点作为块的中心，直到找到包含足够多点的块
        while True:
            # 随机选择一个点作为中心(只取xyz)
            center = points[np.random.choice(point_num)][:3]
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
        current_points = np.zeros((self.num_point, 9))  # 初始化输出数组，num_point * 9

        # 添加全局归一化坐标(相对于整个房间)
        current_points[:, 6] = selected_points[:, 0] / self.data_coord_max[data_id][0]
        current_points[:, 7] = selected_points[:, 1] / self.data_coord_max[data_id][1]
        current_points[:, 8] = selected_points[:, 2] / self.data_coord_max[data_id][2]

        # 坐标中心化(相对于块中心)
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]

        # 颜色归一化(0-255 -> 0-1)
        selected_points[:, 3:6] /= 255.0

        # 将处理后的点云数据填充到输出数组的前6个通道
        current_points[:, 0:6] = selected_points

        # 获取对应的标签
        current_labels = labels[select_point_ids]

        # 应用数据增强变换(如果提供)
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)

        return current_points, current_labels

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
                data = np.loadtxt(os.path.join(data_dir, "datas", name))
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
        class_labels = {'beam': 0, 'board': 1, 'bookcase': 2, 'ceiling': 3, 'chair': 4, 'clutter': 5,
                        'column': 6, 'door': 7, 'floor': 8, 'sofa': 9, 'stairs': 10, 'table': 11, 'wall': 12, 'window': 13}
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
    calculate_dataset_infos(data_dir="../../datasets/st3d", num_classes=14, mode='val')
    # dataset = PCDDataset(data_dir="../../datasets/st3d", num_classes=14)
    # print(len(dataset))
    # for i in range(10):
    #     points, labels = dataset[i]
    #     print(points.shape, labels.shape)


    #
    # data_dir = r"G:\zhongliangjian\my_project\dnn_point_cloud\datasets\Stanford3dDataset_v1.2"
    # output_dir = r"G:\zhongliangjian\my_project\dnn_point_cloud\datasets\st3d\datas"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # areas = [area for area in os.listdir(data_dir) if "Area" in area]
    # labels = set()
    # for area in areas:
    #     print(area)
    #     offices = [office for office in os.listdir(os.path.join(data_dir, area)) if "office" in office]
    #     for office in tqdm(offices, total=len(offices)):
    #         filenames = [name for name in os.listdir(os.path.join(data_dir, area, office, "Annotations")) if name.endswith("txt")]
    #         # for name in filenames:
    #         #     labels.add(name.split("_")[0])
    # # dict_labels = {key: value for value, key in enumerate(sorted(labels))}
    # # print(dict_labels)
    #
    #         labels = None
    #         points = None
    #         for name in filenames:
    #             try:
    #                 data = np.loadtxt(os.path.join(os.path.join(data_dir, area, office, "Annotations"), name))
    #             except Exception as e:
    #                 print("load error: ", name)
    #                 continue
    #             if points is None:
    #                 points = data
    #             else:
    #                 points = np.concatenate([points, data], axis=0)
    #
    #             class_name = name.split('_')[0]
    #             class_id = class_labels[class_name]
    #             label = np.zeros(shape=(data.shape[0], 1), dtype=np.int64)
    #             label[:, :] = class_id
    #             if labels is None:
    #                 labels = label
    #             else:
    #                 labels = np.concatenate([labels, label], axis=0)
    #         data = np.concatenate([points, labels], axis=1)
    #         save_path = os.path.join(output_dir, area + "_" + office + ".txt")
    #         np.savetxt(save_path, data, fmt=["%.4f"] * points.shape[1] + ["%d"], delimiter=' ')

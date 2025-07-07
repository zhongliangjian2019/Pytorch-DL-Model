"""
@brief 数据集
"""
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from dataset_augment import DataTransform

class ClassDataset(Dataset):
    """分类数据集"""
    def __init__(self, data_file: str,
                 in_channel: int = 3,
                 in_size: int = 224,
                 mode: str = 'train',
                 is_transform: bool = True):
        super().__init__()

        assert os.path.exists(data_file), "data_file is not exits"
        self.data_file = data_file

        assert mode in ['train', 'val', 'test'], 'data load mode exception, mode should is "train", "val" or "test"'
        self.mode = mode

        # 从文件解析标签和路径
        self.img_paths = []
        self.cls_labels = []
        with open(data_file, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                label, img_path = line.split(';')
                self.cls_labels.append(int(label))
                self.img_paths.append(img_path.rstrip('\n'))

        self.in_channel = in_channel
        self.size = (in_size, in_size)

        self.transformer = None
        if is_transform:
            self.transformer = DataTransform(p=0.5, input_size=in_size)

    def __len__(self):
        return len(self.cls_labels)

    def __getitem__(self, index: int):
        img = self.loadImage(self.img_paths[index])
        img = cv2.resize(img, dsize=self.size)

        if self.mode == 'train' and self.transformer is not None:
            img = self.transformer(img)

        img = self.preprocess(img)

        label = self.cls_labels[index]

        data = {"image": torch.as_tensor(img, dtype=torch.float32),
                "label": torch.as_tensor(label, dtype=torch.int64)}

        return data

    def loadImage(self, path):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
        if self.in_channel == 3:
            dst = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return dst

    @staticmethod
    def preprocess(src):
        """数据预处理"""
        # 图像缩放到输入大小
        dst = src.astype(np.float32) / 255.0
        if dst.ndim == 2:
            dst = np.expand_dims(dst, axis=0)
        else:
            dst = np.transpose(dst, axes=(2, 0, 1))
        return dst


if __name__ == "__main__":
    """模块测试"""
    pass


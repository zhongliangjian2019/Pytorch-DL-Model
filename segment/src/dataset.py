"""
@brief: 分割模型训练数据加载
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import os.path as pth
import os
from PIL import Image
from typing import Tuple
import tool_func as tf
from dataset_augment import DataTransform
import cv2

class SegmentDataset(Dataset):
    """make数据"""
    def __init__(self, data_dir: str,
                 mode: str = 'train',
                 in_channel: int = 3,
                 num_classes: int = 1,
                 image_size: Tuple[int, int] = (512, 512)):
        super().__init__()
        self.data_dir = data_dir
        self.img_dir = pth.join(data_dir, "images")
        self.mask_dir = pth.join(data_dir, "masks")
        self.ids = self.GetFilenames(os.path.join(data_dir, mode + '.txt'))
        self.num_classes = num_classes
        self.image_size = image_size
        self.in_channel = in_channel

        if mode == 'train':
            self.transform = DataTransform(p=0.5, width=image_size[0], height=image_size[1])
        else:
            self.transform = None

    def GetFilenames(self, txt_file: str):
        """从文本文件获取图像文件名"""
        assert os.path.exists(txt_file), "{} not found".format(txt_file)
        filenames = []
        with open(txt_file, 'r', encoding='utf-8') as file:
            line = file.readline()
            while line:
                filenames.append(line.strip())
                line = file.readline()
        return filenames

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # 读取图像
        filename = self.ids[index]
        img_path = pth.join(self.img_dir, filename)
        if self.in_channel == 3:
            mode = 'RGB'
        else:
            mode = 'L'
        img = self.LoadImage(img_path, mode)

        # 读取标注掩膜
        mask_path = pth.join(self.mask_dir, filename.replace(filename.split('.')[-1], 'png'))
        mask = self.LoadImage(mask_path, mode='P')

        # 图像增强
        if self.transform is not None:
            img, mask = self.transform(image=img, mask=mask)

        # 按长边缩放图像到模型输入尺寸
        img = tf.FormatToSquare(img)
        mask = tf.FormatToSquare(mask)
        img = cv2.resize(img, dsize=(self.image_size[1], self.image_size[0]))
        mask = cv2.resize(mask, dsize=(self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)

        # 数据归一化
        img = img.astype(np.float32) / 255.0
        image = np.transpose(img, (2, 0, 1))

        data = {"image": torch.as_tensor(image, dtype=torch.float32),
                "mask": torch.as_tensor(mask, dtype=torch.int64)}
        return data

    def GetDataID(self, index):
        """获取数据ID"""
        return self.ids[index]

    def LoadImage(self, path, mode: str = 'RGB'):
        """读取图像"""
        img = Image.open(path).convert(mode)
        img = np.array(img)
        return img

if __name__ == "__main__":
    """模块测试"""
    data_dir = "../data"
    dataset = SegmentDataset(data_dir)
    for i in range(len(dataset)):
        data = dataset[i]
        image = data['image']
        mask = data['ground_truth']
        image = image.numpy()
        image = np.squeeze(image) * 255
        image = image.astype(np.uint8)
        mask = mask.numpy() * 100
        mask = mask.astype(np.uint8)
        tf.SaveImage(image, os.path.join(data_dir, 'image_' + str(i) + '.jpg'))
        tf.SaveImage(mask, os.path.join(data_dir, 'mask_' + str(i) + '.png'))
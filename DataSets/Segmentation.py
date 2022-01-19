'''语义分割任务数据加载类'''
import torch
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from DataProcess.BasicProcess import normalization, subtract_mean_value


class SegmentDataset(Dataset):
    '''语义分割任务数据加载类'''

    def __init__(self, root_dir: str = None,
                       image_dir: str = None,
                       mask_dir: str = None,
                       path: str = None,
                       image_format: str = '.jpg',
                       mask_format: str = '.png',
                       transform: Compose = None):
        '''初始化
        :param root_dir: 数据集根目录
        :param image_dir: 图像文件目录（根目录下的相对路径）
        :param mask_dir: 掩膜文件路径（根目录下的相对路径）
        :param image_format: 图像文件格式（后缀名）
        :param mask_format: 掩膜文件格式（后缀名）
        :param path: 文件名id提取路径（根目录下的相对路径）
        '''
        super(SegmentDataset, self).__init__()
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.path = os.path.join(root_dir, path)
        self.name_ids = self.get_name_ids(self.path)  # 文件名id列表
        self.image_format = image_format
        self.mask_format = mask_format

    def __len__(self):
        return len(self.name_ids)

    def __getitem__(self, id):
        # 入参检查
        assert id < len(self.name_ids), "id > max_len, please check id value"

        # 获取数据路径
        image_path = os.path.join(self.root_dir, self.image_dir, self.name_ids[id] + self.image_format)
        mask_path = os.path.join(self.root_dir, self.mask_dir, self.name_ids[id] + self.mask_format)

        # 数据读取
        image = self.load_image(image_path)
        mask = self.load_mask(mask_path)

        # 数据预处理
        image = self.preprocess(image)
        mask = self.preprocess(mask, is_mask=True)

        return {'image': torch.as_tensor(image, dtype=torch.float32),
                'mask': torch.as_tensor(mask, dtype=torch.int64)}

    @classmethod
    def get_name_ids(cls, path: str = None):
        '''获取文件名id
        :param path: 文件路径或文件目录
        :return name_ids: 文件名id列表
        '''
        # 入参检查
        assert path is not None, "path is empty, please check input path"

        # 路径为目录的处理
        if os.path.isdir(path) is True:
            name_ids = [os.path.splitext(name)[0] for name in os.listdir(path) if not os.path.isdir(name)]
            return name_ids

        # 路径为文件的处理
        if os.path.isfile(path) is True:
            with open(path, mode='r', encoding='utf-8') as file:
                name_ids = [str_id.strip('\n') for str_id in file.readlines()]
            return name_ids

        # 路径异常处理(路径既不是文件，也不是目录)
        raise ValueError('path is not valid file path, please check input path')

    @classmethod
    def load_image(cls, path: str = None):
        '''从指定路径加载图像
        :param path: 文件路径
        :return image: 读取的图像（np.ndarray）
        '''
        assert path is not None, "path is empty, please check path"
        image = Image.open(path)
        if image.mode == 'RGBA':
            image = image.convert(mode='RGB')
        image = np.asarray(image)
        return image

    @classmethod
    def load_mask(cls, path):
        '''从指定路径加载图像
        :param path: 文件路径
        :return mask: 读取的掩膜图像（np.ndarray）
        '''
        assert path is not None, "path is empty, please check path"
        mask = Image.open(path)
        mask = mask.convert(mode='P')
        mask = np.asarray(mask)
        return mask

    @classmethod
    def preprocess(cls, input_array: np.ndarray, is_mask: bool = False):
        '''数据预处理
        :param input_array: 输入数据(2D或3D数组)
        :param is_mask: 掩膜数据标志（判断输入数据是否为掩膜）
        :return output_array: 输出数据
        '''
        output_array = np.copy(input_array)
        if is_mask is True:
            output_array[input_array == 255] = 0
        else:
            output_array = normalization(input_array, method='minmax')
            output_array = subtract_mean_value(output_array, mode='pixel')
            if output_array.ndim == 2:
                output_array = np.expand_dims(output_array, axis=-1)
            output_array = np.transpose(output_array, axes=(2, 0, 1))

        return output_array


if __name__ == '__main__':
    '''单元测试'''
    root_dir = '../Data/VOC2007'
    image_dir = 'JPEGImages'
    mask_dir = 'SegmentationClass'
    path = 'ImageSets/Segmentation/trainval.txt'
    my_dataset = SegmentDataset(root_dir, image_dir, mask_dir, path)
    for id in range(len(my_dataset)):
        data = my_dataset[id]
        image = data['image']
        mask = data['mask']
        print(image.shape, mask.shape)
        break
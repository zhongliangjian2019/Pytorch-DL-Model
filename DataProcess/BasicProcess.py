"""
@Description: 基础处理
@Author: zhongliangjian
@Edit Time: 2022/1/19 18:12
"""
import numpy as np


def normalization(input_array: np.ndarray, method: str = 'max'):
    '''数据归一化
    :param input_array: 输入数据，2D或3D矩阵
    :param method: 归一化方法 method = {'max'(default), 'minmax'}
    :return output_array: 归一化结果数据
    '''

    output_array = np.zeros(input_array.shape, dtype=np.float)

    methods = {'max', 'minmax'}
    if method not in methods:
        method = 'max'

    if method == 'max':
        # 最大值归一化
        if input_array.ndim == 2:
            output_array = input_array / np.max(input_array)

        if input_array.ndim == 3:
            for channel in range(input_array.shape[2]):
                output_array[:, :, channel] = input_array[:, :, channel] / np.max(input_array[:, :, channel])

    if method == 'minmax':
        # 最小-最大值归一化
        if input_array.ndim == 2:
            output_array = (input_array - np.min(input_array)) / (np.max(input_array) - np.min(input_array))

        if input_array.ndim == 3:
            for channel in range(input_array.shape[2]):
                output_array[:, :, channel] = (input_array[:, :, channel] - np.min(input_array[:, :, channel])) / \
                                      (np.max(input_array[:, :, channel]) - np.min(input_array[:, :, channel]))

    return output_array


def subtract_mean_value(input_array: np.ndarray, mode: str = 'pixel'):
    '''去均值处理
    :param input_array: 输入数据(2D或3D数组)
    :param mode: 处理模式，mode={'pixel', 'channel'}
    :return output_array: 输出数据
    '''

    output_array = np.zeros(input_array.shape, dtype=np.float)

    if input_array.ndim == 2:
        mode = 'pixel'

    if mode == 'pixel':
        if input_array.ndim == 2:
            output_array = input_array - np.mean(input_array)
        if input_array.ndim == 3:
            for channel in range(input_array.shape[2]):
                output_array[:, :, channel] = input_array[:, :, channel] - np.mean(input_array[:, :, channel])

    if mode == 'channel':
        for channel in range(input_array.shape[2]):
            output_array[:, :, channel] = input_array[:, :, channel] - np.mean(input_array, axis=-1)

    return output_array


if __name__ == '__main__':
    "单元测试"
    from DataSets.Segmentation import SegmentDataset
    import cv2

    root_dir = '../Data/VOC2007'
    image_dir = 'JPEGImages'
    mask_dir = 'SegmentationClass'
    path = 'ImageSets/Segmentation/trainval.txt'
    my_dataset = SegmentDataset(root_dir, image_dir, mask_dir, path)
    for id in range(len(my_dataset)):
        data = my_dataset[id]
        image = data['image']
        print(image.min(), image.max())
        image1 = normalization(image, method='max')
        image2 = normalization(image, method='minmax')
        print('max', image1.min(), image1.max())
        print('minmax', image2.min(), image2.max())
        image3 = subtract_mean_value(image1, mode='pixel')
        image4 = subtract_mean_value(image1, mode='channel')
        print('pixel', image3.min(), image3.max())
        print('channel', image4.min(), image4.max())

        break
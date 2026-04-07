"""
@Description: 模型训练数据加载
@Author: zhongliangjian
@Edit Time: 2022/4/26 19:00
"""

from torch.utils.data import Dataset
import cv2
import numpy as np
import os.path as pth
import os
from PIL import Image
from data_augment import DataTransform

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

class MyDataSet(Dataset):
    """make数据"""
    def __init__(self, data_dir: str,
                 mode: str = 'train',
                 in_channel: int = 3,
                 num_classes: int = 4,
                 image_size: tuple = (512, 512)):
        super(MyDataSet, self).__init__()
        self.data_dir = data_dir
        self.image_dir = pth.join(data_dir, "images")
        self.label_dir = pth.join(data_dir, "labels")
        self.ids = self.GetFilenames(os.path.join(data_dir, mode + '.txt'))
        self.num_classes = num_classes
        self.image_size = image_size
        self.in_channel = in_channel

        if mode == 'train':
            self.transform = DataTransform(p=0.75, width=image_size[0], height=image_size[1])
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

    def GetKeyPoints(self, txt_file: str):
        """从文本文件获取图像文件名"""
        assert os.path.exists(txt_file), "{} not found".format(txt_file)
        with open(txt_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        keypoints = []
        class_labels = []
        for line in lines:
            part = line.strip().split(' ')
            if len(part) < 3:
                continue
            keypoints.append((float(part[1]), float(part[2])))
            class_labels.append(int(part[0]))
        return keypoints, class_labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # 读取图像
        filename = self.ids[index]
        img_path = pth.join(self.image_dir, filename)
        if self.in_channel == 3:
            mode = 'RGB'
        else:
            mode = 'L'
        img = self.LoadImage(img_path, mode)

        # 解析标注数据
        keypoints, class_labels = self.GetKeyPoints(os.path.join(self.label_dir, filename.replace(filename.split('.')[-1], 'txt')))
        if len(keypoints) != 0:
            keypoints = np.array(keypoints)
            keypoints[:, 0] *= img.shape[1]
            keypoints[:, 1] *= img.shape[0]
            keypoints = keypoints.tolist()
            # 图像增强
            if self.transform is not None:
                img, keypoints, class_labels = self.transform(image=img, keypoints=keypoints, class_labels=class_labels)

        # 按长边缩放图像到模型输入尺寸
        height, width = img.shape[0], img.shape[1]
        length = max(height, width)
        image = np.zeros(shape=(length, length, self.in_channel), dtype=np.uint8)
        image[0: height, 0: width, :] = img.copy()
        resize_img = cv2.resize(image, self.image_size)
        resize_img = resize_img.astype(np.float32) / 255.0
        image = np.transpose(resize_img, (2, 0, 1))

        if len(keypoints) != 0:
            keypoints = np.array(keypoints).reshape(-1, 2)
            keypoints[:, 0] = keypoints[:, 0] / length * self.image_size[0]
            keypoints[:, 1] = keypoints[:, 1] / length * self.image_size[1]
            keypoints = keypoints.tolist()
        batch_hm, batch_reg, batch_reg_mask = self.get_pred_encode(keypoints, class_labels)

        return image, batch_hm, batch_reg, batch_reg_mask

    def get_pred_encode(self, keypoints, class_labels: list):
        self.output_shape = self.image_size
        self.input_shape = self.image_size

        batch_hm = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        batch_reg = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg_mask = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)

        window_size = 2
        if len(keypoints) != 0:
            keypoints = np.array(keypoints, dtype=np.float32)
            keypoints[:, 0] = np.clip(keypoints[:, 0] / self.input_shape[1] * self.output_shape[1], window_size,
                                       self.output_shape[1] - window_size - 1)
            keypoints[:, 1] = np.clip(keypoints[:, 1] / self.input_shape[0] * self.output_shape[0], window_size,
                                       self.output_shape[0] - window_size - 1)

        for i in range(len(keypoints)):
            point = keypoints[i].copy()
            cls_id = class_labels[i]
            cls_id = int(cls_id)
            if self.num_classes == 1:
                cls_id = 0
            radius = 11
            # -------------------------------------------------#
            #   计算真实框所属的特征点
            # -------------------------------------------------#
            ct = point
            ct_int = point.astype(np.int32)
            # ----------------------------#
            #   绘制高斯热力图
            # ----------------------------#
            batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)
            # ---------------------------------------------------#
            #   计算中心偏移量
            # ---------------------------------------------------#
            batch_reg[ct_int[1], ct_int[0]] = ct - ct_int
            # ---------------------------------------------------#
            #   将对应的mask设置为1
            # ---------------------------------------------------#
            batch_reg_mask[ct_int[1], ct_int[0]] = 1

        return batch_hm, batch_reg, batch_reg_mask

    def GetDataID(self, index):
        """获取数据ID"""
        return self.ids[index]

    def LoadImage(self, path, mode: str = 'RGB'):
        """读取图像"""
        img = Image.open(path).convert(mode)
        img = np.array(img)
        return img

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data_dir = r'../data/20250324'
    dataset = MyDataSet(data_dir, in_channel=3, num_classes=4, image_size=(512, 512), mode='train')
    for i in range(len(dataset)):
        data = dataset[i]
        image, batch_hm, batch_reg, batch_reg_mask = data
        image = np.squeeze(image) * 255.0
        image = image.astype(np.uint8)
        image = np.transpose(image, axes=[1, 2, 0])
        show_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(show_image)
        plt.show()
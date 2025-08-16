"""
@brief 数据集加载
"""
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from PIL import Image
from data_augment import DataTransform
import math

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_radius(det_size: tuple, min_overlap: float = 0.7):
    """计算高斯半径"""
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
    """数据加载"""
    def __init__(self, data_dir: str,
                 mode: str = 'train',
                 in_channel: int = 3,
                 num_classes: int = 1,
                 image_size: tuple = (512, 512)):
        super(MyDataSet, self).__init__()
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images", mode)
        self.label_dir = os.path.join(data_dir, "labels", mode)
        self.filenames = [name for name in os.listdir(self.image_dir) if name.split(".")[-1] in ("jpg", "bmp", "tif")]
        self.num_classes = num_classes
        self.input_shape = image_size
        self.in_channel = in_channel
        self.transform = None

        if mode == 'train':
            self.transform = DataTransform(p=0.5, width=image_size[0], height=image_size[1])

    def __len__(self):
        return len(self.filenames)

    def parse_yolo_label_file(self, txt_file: str):
        """读取YOLO标签文件"""
        bboxes = []
        class_labels = []
        if os.path.exists(txt_file):
            with open(txt_file, 'r') as file:
                lines = file.readlines()
            for line in lines:
                line = line.strip("\n")
                id, cx, cy, w, h = line.split(" ")
                bboxes.append([float(cx), float(cy), float(w), float(h)])
                class_labels.append(id)
        return bboxes, class_labels

    def load_image(self, path: str, mode: str = 'RGB'):
        """读取图像"""
        return np.array(Image.open(path).convert(mode))

    def __getitem__(self, index: int):
        # 读取图像
        filename = self.filenames[index]
        image_file = os.path.join(self.image_dir, filename)
        mode = "RGB" if self.in_channel == 3 else "L"
        image = self.load_image(image_file, mode)

        # 读取标注掩膜
        label_file = os.path.join(self.label_dir, filename.replace(filename.split(".")[-1], "txt"))
        bboxes, class_labels = self.parse_yolo_label_file(label_file)

        if len(bboxes) != 0:
            bboxes = np.array(bboxes, dtype=np.float32)
            class_labels = np.array(class_labels)
            # 数据增强
            if self.transform is not None:
                image, bboxes, class_labels = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)

        # 填充图像到输入尺寸
        scale = min(self.input_shape[0] / image.shape[0], self.input_shape[1] / image.shape[1])
        new_height = min(int(image.shape[0] * scale), self.input_shape[0])
        new_width = min(int(image.shape[1] * scale), self.input_shape[1])
        input_image = np.zeros(shape=(self.input_shape[0], self.input_shape[1], self.in_channel), dtype=np.uint8)
        if self.in_channel == 1:
            input_image[0:new_height, 0:new_width, 0] = cv2.resize(image, dsize=(new_width, new_height))
        else:
            input_image[0:new_height, 0:new_width, :] = cv2.resize(image, dsize=(new_width, new_height))
        input_image = input_image.astype(np.float32)
        input_image /= 255.0
        input_image = np.transpose(input_image, axes=(2, 0, 1))

        # 同步bboxes尺寸
        if len(bboxes) != 0:
            bboxes[:, [0, 2]] *= new_width
            bboxes[:, [1, 3]] *= new_height
            bboxes[:, [0, 2]] /= self.input_shape[1]
            bboxes[:, [1, 3]] /= self.input_shape[0]

        # 按长边缩放图像到模型输入尺寸
        ground_truth = self.get_ground_truth(bboxes, class_labels)

        return input_image, ground_truth

    def get_ground_truth(self, bboxes: np.ndarray, class_labels: np.ndarray):
        """获取真实标签"""
        # 模型输出分辨率
        self.output_shape = self.input_shape
        # 模型输出真值
        cla_hm     = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        reg_wh     = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        reg_offset = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        reg_mask   = np.zeros((self.output_shape[0], self.output_shape[1], 1), dtype=np.float32)
        # 还原bbox到输出尺度
        if len(bboxes) != 0:
            bboxes = bboxes.astype(np.float32)
            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]] * self.output_shape[1], 0, self.output_shape[1] - 1)
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]] * self.output_shape[0], 0, self.output_shape[0] - 1)

        for bbox, label in zip(bboxes, class_labels):
            cx, cy, w, h = bbox
            class_id = int(label)
            if h > 0 and w > 0:
                # 计算目标高斯半径
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(1, int(radius))
                # 绘制高斯热力图
                center = (int(cx), int(cy))
                cla_hm[:, :, class_id] = draw_gaussian(cla_hm[:, :, class_id], center, radius)
                # 高宽预测
                reg_wh[max(0, center[1] - radius) : min(center[1] + radius + 1, self.output_shape[0]),
                       max(0, center[0] - radius) : min(center[0] + radius + 1, self.output_shape[1])] = w * 0.1, h * 0.1
                # 中心量化误差预测
                reg_offset[max(0, center[1] - radius): min(center[1] + radius + 1, self.output_shape[0]),
                           max(0, center[0] - radius): min(center[0] + radius + 1, self.output_shape[1])] = cx - center[0], cy - center[1]
                # 回归掩膜
                reg_mask[max(0, center[1] - radius): min(center[1] + radius + 1, self.output_shape[0]),
                         max(0, center[0] - radius): min(center[0] + radius + 1, self.output_shape[1])] = 1

        return np.concatenate([cla_hm, reg_wh, reg_offset, reg_mask], axis=-1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from model_inference import decode_bbox
    data_dir = "..\data"
    dataset = MyDataSet(data_dir, in_channel=3, num_classes=1, image_size=(512, 512), mode='train')
    for i in range(len(dataset)):
        data = dataset[i]
        image, ground_truth = data
        image = np.transpose(image, axes=(1, 2, 0))
        cla_hm = ground_truth[:, :, 0]
        cla_hm = np.squeeze(cla_hm)
        reg_wh = ground_truth[:, :, 1:3]
        reg_offset = ground_truth[:, :, 3:5]
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title(label="Image")
        plt.subplot(1, 3, 2)
        plt.imshow(cla_hm, cmap='jet')
        plt.title(label="Heatmap")
        plt.subplot(1, 3, 3)
        plt.imshow(reg_wh[:, :, 0], cmap='jet')
        plt.title(label="Radius")
        plt.show()
        batch_hm = cla_hm[np.newaxis, np.newaxis, ...]
        batch_wh = np.transpose(reg_wh, axes=[2, 0, 1])
        batch_wh = batch_wh[np.newaxis, ...]
        batch_reg = np.transpose(reg_offset, axes=[2, 0, 1])
        batch_reg = batch_reg[np.newaxis, ...]
        detects = decode_bbox(batch_hm, batch_wh, batch_reg, 0.5)

        image = np.squeeze(image) * 255.0
        image = image.astype(np.uint8)
        # show_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        show_image = image.copy()
        print(show_image.shape)
        for detect in detects:
            if len(detect) != 0:
                detect[:, [0, 2]] *= image.shape[1]
                detect[:, [1, 3]] *= image.shape[0]
                for i in range(detect.shape[0]):
                    box = detect[i, :4].astype(np.int32)
                    score = detect[i, 4]
                    cla_id = detect[i, 5]
                    cv2.rectangle(show_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        plt.imshow(show_image)
        plt.show()
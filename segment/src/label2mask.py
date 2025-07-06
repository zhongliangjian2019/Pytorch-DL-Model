"""转换labelme标注到掩膜图像"""
import cv2
import numpy as np
import os
import json
from tqdm import tqdm
from PIL import Image
import shutil

# 类别ID
labels = {'background': 0, 'line': 1, 'short_line': 2}
# 背景标签类别
background_labels = {'background'}
# 调色板
colormap = [0, 0, 127, 0, 127, 0, 127, 0, 0, 0, 0, 0]

def label_to_mask(input_dir: str, out_dir: str, file_extra: str = 'jpg'):
    """转换标注到掩膜"""
    out_dirs = {'images' : os.path.join(out_dir, 'images'), 'masks': os.path.join(out_dir, 'masks')}
    for sub_dir in out_dirs.values():
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

    image_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.split('.')[-1] == file_extra]

    with tqdm(total=len(image_files), desc='Running:') as pbar:
        for file in image_files:
            json_path = file.replace('.' + file_extra, '.json')
            if os.path.exists(json_path):
                with open(json_path, mode='r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                width, height = data['imageWidth'], data['imageHeight']
                shapes = data['shapes']
                mask = np.zeros((height, width), np.uint8)
                background = np.ones((height, width), np.uint8)
                for shape in shapes:
                    if shape['shape_type'] != 'polygon':
                        continue
                    contour = np.int64(shape['points'])
                    shape_label = shape['label']
                    if shape_label in background_labels:
                        label_id = 0
                        cv2.drawContours(background, [contour], 0, label_id, -1)
                    else:
                        label_id = labels[shape_label]
                        cv2.drawContours(mask, [contour], 0, label_id, -1)
                # 去除背景区域
                mask = mask * background
                # 转换PIL的P编码图像
                image = Image.fromarray(mask)
                image_p = image.convert("P")
                image_p.putpalette(colormap)

                # 保存结果
                src_path = os.path.join(input_dir, data['imagePath'])
                dst_path = os.path.join(out_dirs['images'], data['imagePath'])
                shutil.copyfile(src_path, dst_path)

                image_p.save(os.path.join(out_dirs['masks'], data['imagePath'].replace(data['imagePath'].split('.')[-1], 'png')))
            else:
                image = cv2.imread(file, 0)
                mask = np.zeros_like(image)
                # 转换PIL的P编码图像
                image = Image.fromarray(mask)
                image_p = image.convert("P")
                image_p.putpalette(colormap)
                # 保存结果
                src_path = file
                dst_path = os.path.join(out_dirs['images'], os.path.basename(file))
                shutil.copyfile(src_path, dst_path)

                image_p.save(os.path.join(out_dirs['masks'], os.path.basename(file).replace(os.path.basename(file).split('.')[-1], 'png')))

            pbar.update()

if __name__ == "__main__":
    """单元测试"""
    input_dir = "../data"
    out_dir = input_dir
    label_to_mask(input_dir, out_dir)

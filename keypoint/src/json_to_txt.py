"""
将json的点标注, 转换为类似于yolo标注文件形式的txt文件
json文件为点标注
txt文件每行为：cla_id pt_x pt_y
"""
import json
import os
from tqdm import tqdm

# 关键点类别信息
class_names = {'code': 0}

def main1(src_dir: str, dst_dir: str):
    # 输出目录
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    # 迭代处理
    filenames = [name for name in os.listdir(src_dir) if name.endswith('json')]
    with tqdm(total=len(filenames), desc="running") as pbar:
        for name in filenames:
            json_file = os.path.join(src_dir, name)
            with open(json_file, mode='r', encoding='utf-8') as file:
                data = json.load(file)
            width = data["imageWidth"]
            height = data["imageHeight"]
            shapes = data["shapes"]
            key_points = []
            for shape in shapes:
                if shape['shape_type'] != 'polygon':
                    continue
                cla_id = class_names[shape['label']]
                for point in shape['points']:
                    key_points.append("%d %.6f %.6f\n" % (cla_id, point[0] / width, point[1] / height))
            label_file = os.path.join(output_dir, name.replace('json', 'txt'))
            with open(label_file, mode='w', encoding='utf-8') as file:
                file.writelines(key_points)
            pbar.update()


if __name__ == "__main__":
    # json文件目录
    data_dir = r"Z:\dataset\qrcode_keypoints\images"
    # 输出目录
    output_dir = os.path.join(os.path.dirname(data_dir), "labels")
    main1(data_dir, output_dir)
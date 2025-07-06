"""数据集划分: 分割任务
数据集目录格式：
- data_dir：数据集根目录
    - images: 图像数据
    - masks: 掩膜数据（p编码）
"""
import os
from sklearn.model_selection import train_test_split

def data_split(data_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """数据集划分"""
    # 提取具有标注的图像文件
    image_files = [name for name in os.listdir(os.path.join(data_dir, "images")) if name.split('.')[-1] in ['jpg', 'png', 'bmp', 'tif']]
    mask_files = [name.split('.')[0] for name in os.listdir(os.path.join(data_dir, "masks")) if name.split('.')[-1] == 'png']
    filenames = [name for name in image_files if name.split('.')[0] in mask_files]

    # 计算划分数量
    valid_size = int(len(filenames) * val_ratio)
    test_size = int(len(filenames) * (1 - train_ratio - val_ratio))

    print("total_data: %d, train_data: %d(%.2f), val_data: %d(%.2f), test_data: %d(%.2f)" % (len(filenames),
           len(filenames) - valid_size - test_size, train_ratio,
           valid_size, val_ratio,
           test_size, 1 - train_ratio - val_ratio))

    # 数据划分
    train_files, test_files, train_labels, test_labels = train_test_split(filenames, filenames, test_size=test_size, random_state=0)
    train_files, val_files, _, _ = train_test_split(train_files, train_labels, test_size=valid_size, random_state=0)
    train_datas, val_datas, test_datas = [], [], []
    for file in train_files:
        train_datas.append(file + "\n")
    for file in val_files:
        val_datas.append(file + "\n")
    for file in test_files:
        test_datas.append(file + "\n")

    # 写入文件
    with open(os.path.join(data_dir, "train.txt"), mode='w') as file:
        file.writelines(train_datas)

    with open(os.path.join(data_dir, "val.txt"), mode='w') as file:
        file.writelines(val_datas)

    with open(os.path.join(data_dir, "test.txt"), mode='w') as file:
        file.writelines(test_datas)

    print("data split finished, result output to train.txt, val.txt, test.txt")


if __name__ == '__main__':
    """模块测试"""
    data_dir = "../data"
    data_split(data_dir)



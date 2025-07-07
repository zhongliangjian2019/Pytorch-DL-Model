"""
@brief 数据集划分
"""
import os
from sklearn.model_selection import train_test_split

# 类别标签
label_dicts = {'class1': 0, 'class2': 1, 'class3': 2}

def data_split(data_dir: str, train_ratio: float = 0.8, test_ratio: float = 0.1):
    """数据集划分"""
    sub_dirs = [sub_dir for sub_dir in os.listdir(data_dir) if sub_dir[0] != '.']
    train_datas, val_datas, test_datas = [], [], []
    for sub_dir in sub_dirs:
        filenames = [name for name in os.listdir(os.path.join(data_dir, sub_dir)) if name.split('.')[-1] in ['jpg', 'png', 'bmp']]
        labels = [label_dicts[sub_dir] for i in range(len(filenames))]
        train_files, test_files, train_labels, test_labels = train_test_split(filenames, labels, test_size=test_ratio)
        train_files, val_files, _, _ = train_test_split(train_files, train_labels, test_size= 1 - train_ratio)
        for file in train_files:
            train_datas.append(str(label_dicts[sub_dir]) + ";" + os.path.join(data_dir, sub_dir, file) + "\n")
        for file in val_files:
            val_datas.append(str(label_dicts[sub_dir]) + ";" + os.path.join(data_dir, sub_dir, file) + "\n")
        for file in test_files:
            test_datas.append(str(label_dicts[sub_dir]) + ";" + os.path.join(data_dir, sub_dir, file) + "\n")

    with open(os.path.join(data_dir, "train.txt"), mode='w', encoding='utf-8') as file:
        file.writelines(train_datas)

    with open(os.path.join(data_dir, "val.txt"), mode='w', encoding='utf-8') as file:
        file.writelines(val_datas)

    with open(os.path.join(data_dir, "test.txt"), mode='w', encoding='utf-8') as file:
        file.writelines(test_datas)


if __name__ == '__main__':
    """模块测试"""
    data_dir = r"..\data"
    data_split(data_dir)



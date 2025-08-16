"""
@brief 检测数据集划分
"""
import os
from tqdm import tqdm
import random
import shutil

def split_to_dataset(image_dir: str, label_dir: str, train_ratio: float = 0.9, val_ratio: float = 0.1, img_format: str = 'tif'):
    """划分数据集"""
    idxs = set(name.split('.')[0] for name in os.listdir(image_dir) if name.split('.')[-1] == img_format)

    n_total = len(idxs)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)

    train_idxs = random.sample(idxs, n_train)
    idxs.difference_update(train_idxs)
    val_idxs = random.sample(idxs, n_val)
    test_idxs = list(idxs.difference(val_idxs))

    dir_dict = {'images': image_dir, 'labels': label_dir}
    format_dict = {'images': img_format, 'labels': 'txt'}
    data_dict = {'train': train_idxs, 'val': val_idxs, 'test': test_idxs}

    with tqdm(total=n_total, desc="split dataset: ") as pbar:
        for dir_type in dir_dict.keys():
            for set_type in data_dict.keys():
                os.makedirs(os.path.join(dir_dict[dir_type], set_type))
                for idx in data_dict[set_type]:
                    src = os.path.join(dir_dict[dir_type], idx + '.' + format_dict[dir_type])
                    dst = os.path.join(dir_dict[dir_type], set_type, idx + '.' + format_dict[dir_type])
                    if os.path.exists(src):
                        shutil.move(src, dst)
                    pbar.update()

if __name__ == '__main__':
    data_dir = r".\data"
    split_to_dataset(os.path.join(data_dir, 'images'), os.path.join(data_dir, 'labels'), img_format='jpg')
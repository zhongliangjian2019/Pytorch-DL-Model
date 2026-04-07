"""
@Description: 模型训练
"""
import cv2
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from dataset import MyDataSet
from model import KeyPointModel
from loss_func import focal_loss, reg_l1_loss
import os
import logging
import datetime
import shutil
import argparse
# from decode_keypoint import decode_bbox_cpu
# from wandb_show import wandb_boxes2d

# 日志文件格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class SegConfig:
    """分割模型配置项"""
    def __init__(self):
        self.data_dir = ''                  # 数据加载目录
        self.batch_size = 256               # batch大小
        self.num_classes = 1                # 模型分类数
        self.in_channel = 3                 # 输入图像通道数
        self.image_size = (192, 192)        # 输入图像尺寸
        self.learning_rate = 1e-3           # 初始学习率
        self.epochs = 300                   # 训练epoch数
        self.load_num_workers = 16          # 数据加载线程数
        self.is_wandb = True                # 是否启用wandb可视化
        self.checkpoint = None

def GetDataLoader(cfg: SegConfig):
    """
    功能描述：获取数据加载器
    return: train_loader, n_train, val_loader, n_val
    """
    assert os.path.exists(cfg.data_dir), "find not data directory"

    # 1.创建数据集
    train_set = MyDataSet(cfg.data_dir, mode='train', in_channel=cfg.in_channel, num_classes=cfg.num_classes, image_size=cfg.image_size)
    val_set   = MyDataSet(cfg.data_dir, mode='val', in_channel=cfg.in_channel, num_classes=cfg.num_classes, image_size=cfg.image_size)

    # 2.划分训练集与验证集
    n_train = len(train_set)
    n_val   = len(val_set)

    # 3.创建数据加载器
    loader_args = dict(batch_size=cfg.batch_size, num_workers=cfg.load_num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader   = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    return train_loader, n_train, val_loader, n_val

def ConfigModel(model, cfg: SegConfig):
    """
    功能描述：配置模型（训练设备、损失函数、优化器、学习率及衰减机制）
    """
    # 1.配置训练设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    logging.info("train device: {0}".format(device))

    # 2.损失函数
    # weight = None
    # if cfg.class_weight is not None:
    #     weight = torch.tensor(cfg.class_weight, dtype=torch.float32)
    criterion = None

    # 3.优化器
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-2)

    # 4.学习率监督器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, min_lr=1e-5)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 300, 100)

    return model, device, criterion, optimizer, scheduler

def TrainOneEpoch(model, train_loader, device, criterion, optimizer, epoch, epochs):
    """
    功能描述：单轮次训练
    :param model：网络模型
    :param train_loader: 训练集加载器
    :param device: 训练设备
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param epoch: 迭代轮次
    :param epochs: 迭代总轮次
    :return:
        images：最后一个批次的图像
        labels：最后一个批次的标签
        pred_labels：最后一个批次的预测
        train_loss：训练平均损失
    """
    model.train()
    total_loss = 0
    total_c_loss = 0
    total_r_loss = 0
    try:
        with tqdm(desc="train epoch[{0}/{1}]".format(epoch, epochs), total=len(train_loader)) as pbar:
            for data in train_loader:
                # 加载数据
                batch_images, batch_hms, batch_regs, batch_reg_masks = data
                batch_images = batch_images.to(device)

                hm, offset = model(batch_images)
                hm = hm.cpu()
                offset = offset.cpu()
                c_loss = focal_loss(hm, batch_hms)
                off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                loss = c_loss + off_loss

                total_loss += loss.item()
                total_c_loss += c_loss.item()
                total_r_loss += off_loss.item()

                # 梯度清零
                optimizer.zero_grad(set_to_none=True)
                # 反向传播
                loss.backward()
                # 更新参数
                optimizer.step()

                pbar.set_postfix(all_loss="{0:.4f}".format(loss.item()),
                                 cla_loss="{0:.4f}".format(c_loss.item()),
                                 reg_loss="{0:.4f}".format(off_loss.item()))
                pbar.update()

            if len(train_loader) != 0:
                total_loss /= len(train_loader)
                total_c_loss /= len(train_loader)
                total_r_loss /= len(train_loader)

            pbar.set_postfix(all_loss=total_loss, cla_loss=total_c_loss, reg_loss=total_r_loss,
                             lr=optimizer.param_groups[0]['lr'])
    except KeyboardInterrupt:
        pbar.close()

    return batch_images, hm, offset, total_loss

def Validation(model, device, val_loader, epoch, epochs, criterion=None):
    """
    功能描述：模型验证
    :param model: 网络模型
    :param device: 训练设备
    :param val_loader: 验证集加载器
    :param epoch: 当前迭代轮次
    :param epochs: 总迭代轮次
    :return:
        images：最后一个批次的图像
        labels：最后一个批次的标签
        pred_labels：最后一个批次的预测
        val_loss：平均验证损失
        correct_rate：正确率
    """
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_c_loss = 0
        total_r_loss = 0
        try:
            with tqdm(desc="valid epoch[{0}/{1}]".format(epoch, epochs), total=len(val_loader)) as pbar:
                for data in val_loader:
                    # 加载数据
                    batch_images, batch_hms, batch_regs, batch_reg_masks = data

                    hm, offset = model(batch_images.to(device))
                    hm = hm.cpu()
                    offset = offset.cpu()
                    c_loss = focal_loss(hm, batch_hms)
                    off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                    loss = c_loss + off_loss

                    total_loss += loss.item()
                    total_c_loss += c_loss.item()
                    total_r_loss += off_loss.item()

                    pbar.set_postfix(score="{0:.4f}".format(loss.item()))
                    pbar.update()

                if len(val_loader) != 0:
                    total_loss /= len(val_loader)
                    total_c_loss /= len(val_loader)
                    total_r_loss /= len(val_loader)

                pbar.set_postfix(all_loss=total_loss, cla_loss=total_c_loss, reg_loss=total_r_loss)

        except KeyboardInterrupt:
            pbar.close()

    return batch_images, batch_hms, batch_regs, hm, offset, total_loss

def train_model(cfg: SegConfig):
    """功能描述：模型训练"""
    # 创建模型
    model = KeyPointModel(in_channel=cfg.in_channel, num_classes=cfg.num_classes)

    # 加载预训练权重
    if cfg.checkpoint is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(cfg.checkpoint, map_location='cpu', weights_only=True)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

    # 创建模型保存节点目录
    cur_time = datetime.datetime.now()
    dir_checkpoint = "../checkpoints/ckpt_{0}".format(cur_time.strftime("%Y%m%d-%H%M%S"))
    if os.path.exists(dir_checkpoint):
        shutil.rmtree(dir_checkpoint)
    os.makedirs(dir_checkpoint)

    # 1.获取数据加载器
    train_loader, n_train, val_loader, n_val = GetDataLoader(cfg)

    # 2.配置训练模型
    model, device, criterion, optimizer, scheduler = ConfigModel(model, cfg)

    # 3.初始化训练监测
    experiment = None
    if cfg.is_wandb:
        experiment = wandb.init(project='DPCR', entity='zhongliangjian', name=cur_time.strftime("%Y%m%d-%H%M%S"))
        experiment.config.update(dict(epochs=cfg.epochs, batch_size=cfg.batch_size, learning_rate=cfg.learning_rate))

    # 5.开始训练（训练 + 验证）
    best_val_loss = 1e6
    epochs = cfg.epochs
    for epoch in range(1, cfg.epochs):
        logging.info("epoch: [{0} / {1}]".format(epoch, epochs))

        # 训练
        batch_images, hm, offset, train_loss = TrainOneEpoch(model, train_loader, device, criterion, optimizer, epoch, epochs)

        # 验证
        batch_images, batch_hms, batch_regs, hm, offset, val_loss = Validation(model, device, val_loader, epoch, epochs, criterion)

        # 学习率监督
        scheduler.step(val_loss)

        with torch.no_grad():
            # 保存训练节点
            torch.save(model.state_dict(), os.path.join(dir_checkpoint, "last.pth"))

            # 基础监控信息
            if experiment is not None:
                experiment.log({'learning rate': optimizer.param_groups[0]['lr'],
                                'train_loss': train_loss,
                                'val_loss': best_val_loss,
                                'epoch': epoch})

            # 最佳模型监测
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(dir_checkpoint, "best.pth"))

                # 精度最佳模型监测
                if experiment is not None:
                    images = batch_images.cpu().numpy().transpose([0, 2, 3, 1])
                    p_hms = hm.cpu().numpy()
                    g_hms = batch_hms.cpu().numpy().transpose([0, 3, 1, 2])
                    show_images = []
                    show_count = min(5, images.shape[0])
                    class_num = p_hms.shape[1]
                    for i in range(show_count):
                        image = images[i, :, :, :].copy()
                        image = (image * 255).astype(np.uint8)
                        show_images.append(wandb.Image(image, caption='image_%d' % i))
                        for j in range(class_num):
                            pheatmap = p_hms[i, j, :, :].copy()
                            pheatmap = ((1.0 - pheatmap) * 255).astype(np.uint8)
                            pheatmap = cv2.applyColorMap(pheatmap, cv2.COLORMAP_JET)
                            gheatmap = g_hms[i, j, :, :].copy()
                            gheatmap = ((1.0 - gheatmap) * 255).astype(np.uint8)
                            gheatmap = cv2.applyColorMap(gheatmap, cv2.COLORMAP_JET)
                            show_images.append(wandb.Image(gheatmap, caption="gheatmap_%d:%d" % (i, j)))
                            show_images.append(wandb.Image(pheatmap, caption="pheatmap_%d:%d" % (i, j)))
                    wandb.log({"show_result": show_images})

if __name__ == "__main__":
    # 配置参数
    parser = argparse.ArgumentParser(description="segment model training script")
    parser.add_argument('-b', '--batch_size', type=int, default=256, help="setting batch size, default = 16")
    parser.add_argument('-lr', '--learn_rate', type=float, default=1e-3, help="setting initial learning rate, default = 0.001")
    parser.add_argument('-e', '--epochs', type=int, default=1000, help="setting training total epochs number, default = 1000")
    parser.add_argument('-d', '--data_dir', type=str, default=r'../data/qrcode_keypoints', help="setting training data load directory")
    parser.add_argument('-c', '--num_classes', type=int, default=1, help="setting model classify number, should contain background")
    parser.add_argument('-ch', '--in_channels', type=int, default=3, help="setting model input channel number")
    parser.add_argument('-m', '--model', type=str, default='unet', help="setting model mode, 0 - dawei, 1 - maccura")
    parser.add_argument('-ckp', '--checkpoint', type=str, default="../checkpoints/last.pth", help="setting model load checkpoint")
    args = parser.parse_args()

    cfg = SegConfig()
    cfg.data_dir = args.data_dir
    cfg.batch_size = args.batch_size
    cfg.num_classes = args.num_classes
    cfg.learning_rate = args.learn_rate
    cfg.in_channel = args.in_channels
    cfg.checkpoint = args.checkpoint
    cfg.image_size = (128, 128)
    cfg.epochs = args.epochs
    cfg.is_wandb = True
    cfg.load_num_workers = 8

    # 日志信息
    logging.info("model training configure:")
    logging.info("data_dir: {0}".format(cfg.data_dir))
    logging.info("in_channels: {0}".format(cfg.in_channel))
    logging.info("num_classes: {0}".format(cfg.num_classes))
    logging.info("batch_size: {0}".format(cfg.batch_size))
    logging.info("learn_rate: {0}".format(cfg.learning_rate))
    logging.info("iter_epochs: {0}".format(cfg.epochs))
    logging.info("checkpoint: {0}".format(cfg.checkpoint))

    train_model(cfg)

"""
@brief 模型训练
"""
from pointnet import PointNetSegmenter
from pointnet2_seg import PointNet2SegMSG
from pointcnn_seg import PointCNNSegmenter
from dataset import PCDDataset
from loss_func import LossFunc
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from datetime import datetime
import logging
import argparse

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

np.random.seed(0)
colors = np.random.randint(0, 255, size=[14, 3])
colors = torch.as_tensor(colors, dtype=torch.uint8)

class PCDModelTrainer:
    """点云处理模型训练器"""
    def __init__(self, model_name: str, data_dir: str, ckpt_dir: str, ckpt_file: str = None, log_dir: str = None,
                 in_point_dim: int = 3, in_feat_dim: int = 9, sampled_point_num: int = 2048, num_classes: int = 14, batch_size: int = 32, device: int = 0,
                 lr: float = 1e-3, epochs: int = 100):
        """
        :param model_name: 模型名, {'PointNet', 'PointCNN', 'PointNet2'}
        :param data_dir: 数据目录
        :param ckpt_dir: 模型检查点保存目录
        :param ckpt_file: 预训练权重文件名
        :param log_dir: 训练日志保存目录
        :param in_point_dim: 输入点的空间维度2D/3D
        :param in_feat_dim: 输入点的特征维度, 如 rgb/norm_xyz
        :param num_classes: 分类/分割类别数
        :param batch_size: 批处理大小
        :param device: 设备ID
        :param lr: 学习率
        :param epochs: 训练时代数
        """
        self.epochs = epochs
        self.model = None
        self.device = device
        self.data_dir = data_dir
        self.ckpt_dir = ckpt_dir
        self.ckpt_file = ckpt_file
        self.in_point_dim = in_point_dim
        self.in_feat_dim = in_feat_dim
        self.sampled_point_num = sampled_point_num
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.log_dir = log_dir
        if self.log_dir is None:
            self.log_dir = "./runs/run-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.logger = logging.getLogger("trainer")
        self.model_name = model_name

        # 输出日志信息
        self.logger.info("Trainer configure:")
        self.logger.info(f"model_name: {self.model_name}")
        self.logger.info(f"data_dir: {self.data_dir}")
        self.logger.info(f"ckpt_dir: {self.ckpt_dir}")
        self.logger.info(f"ckpt_file: {self.ckpt_file}")
        self.logger.info(f"log_dir: {self.log_dir}")
        self.logger.info(f"num_classes: {self.num_classes}")
        self.logger.info(f"epochs: {self.epochs}")
        self.logger.info(f"batch_size: {self.batch_size}")
        self.logger.info(f"lr: {self.lr}")
        self.logger.info(f"device: {self.device}")

    def train_one_epoch(self, epoch: int, train_loader: DataLoader, criterion: LossFunc, optimizer: Optimizer):
        """
        进行一轮训练
        :param model: 模型
        :param epoch: 轮次标识
        :param train_datas: 训练数据
        :param criterion: 损失函数
        :return:
        """
        self.model.train()
        mean_loss, mean_c_loss, mean_t_loss = 0, 0, 0
        with tqdm(total=len(train_loader), desc=f"train-[{epoch}/{self.epochs}]") as pbar:
            for iter, data in enumerate(train_loader):
                inputs, labels = data
                inputs = inputs.to(device=self.device)
                labels = labels.to(device=self.device)

                if self.model_name == "PointNet":
                    outputs, feat_trans = self.model(inputs)
                else:
                    # PointCNN/PointNet++
                    outputs = self.model(inputs)
                    feat_trans = None

                loss, c_loss, t_loss = criterion(outputs, labels, feat_trans)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                mean_loss += loss.item()
                mean_c_loss += c_loss.item()
                mean_t_loss += t_loss.item()

                self.writer.add_scalars("train_loss",
                                   dict(loss=loss.item(), c_loss=c_loss.item(), t_loss=t_loss.item()),
                                   global_step=iter + epoch * len(train_loader))

                pbar.set_postfix(dict(iter=iter, loss=loss.item(), c_loss=c_loss.item(), t_loss=t_loss.item()))
                pbar.update()

        mean_loss /= len(train_loader)
        mean_c_loss /= len(train_loader)
        mean_t_loss /= len(train_loader)

        return (mean_loss, mean_c_loss, mean_t_loss)

    def validation_one_epoch(self, epoch: int, val_loader: DataLoader, criterion: LossFunc):
        """
        进行一轮验证
        :param epoch:
        :param epochs:
        :param val_loader:
        :param model:
        :param criterion:
        :param device:
        :return:
        """
        self.model.eval()
        with torch.no_grad():
            mean_loss, mean_c_loss, mean_t_loss = 0, 0, 0
            with tqdm(total=len(val_loader), desc=f"val-[{epoch}/{self.epochs}]") as pbar:
                for iter, data in enumerate(val_loader):
                    inputs, labels = data
                    inputs = inputs.to(device=self.device)
                    labels = labels.to(device=self.device)

                    if self.model_name == "PointNet":
                        outputs, feat_trans = self.model(inputs)
                    else:
                        outputs = self.model(inputs)
                        feat_trans = None

                    loss, c_loss, t_loss = criterion(outputs, labels, feat_trans)

                    mean_loss += loss.item()
                    mean_c_loss += c_loss.item()
                    mean_t_loss += t_loss.item()

                    self.writer.add_scalars("val_loss",
                                       dict(loss=loss.item(), c_loss=c_loss.item(), t_loss=t_loss.item()),
                                       global_step=iter + epoch * len(val_loader))

                    pbar.set_postfix(dict(iter=iter, loss=loss.item(), c_loss=c_loss.item(), t_loss=t_loss.item()))
                    pbar.update()

                mean_loss /= len(val_loader)
                mean_c_loss /= len(val_loader)
                mean_t_loss /= len(val_loader)

            return (inputs, labels, outputs), (mean_loss, mean_c_loss, mean_t_loss)

    def run(self):
        # 1.准备数据
        # 训练集
        train_datas = PCDDataset(self.data_dir, self.num_classes, mode='train',
                                 sample_point_num=self.sampled_point_num, sample_block_size=1.0, transform=True)
        train_loader = DataLoader(train_datas, self.batch_size, shuffle=True, num_workers=8,
                                  pin_memory=True, drop_last=True)
        # 验证集
        val_datas = PCDDataset(self.data_dir, self.num_classes, mode='val',
                               sample_point_num=self.sampled_point_num, sample_block_size=1.0, transform=False)
        val_loader = DataLoader(val_datas, self.batch_size, shuffle=False, num_workers=8,
                                pin_memory=True, drop_last=False)

        # 2.加载模型与配置
        self.model = None
        if self.model_name == "PointNet":
            self.model = PointNetSegmenter(point_dims=self.in_point_dim,
                                           feat_dims=self.in_feat_dim,
                                           class_num=self.num_classes,
                                           feat_scale=32)
        if self.model_name == "PointCNN":
            self.model = PointCNNSegmenter(point_nums=self.sampled_point_num,
                                           space_dims=self.in_point_dim,
                                           in_features=self.in_feat_dim,
                                           num_classes=self.num_classes,
                                           feature_scale=32,
                                           ksize=32)
        if self.model_name == "PointNet2":
            self.model = PointNet2SegMSG(point_dims=self.in_point_dim,
                                         feat_dims=self.in_feat_dim,
                                         num_classes=self.num_classes)
        if self.model is None:
            self.logger.error(f"{self.model_name} is not found matched model")
            return

        # 加载预训练参数
        if self.ckpt_file is not None:
            self.model.load_state_dict(torch.load(self.ckpt_file), strict=False)
        self.model.to(device=self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = LossFunc(train_datas.class_weights, trans_loss_scale=0.001)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=5, cooldown=1, min_lr=1e-6)

        best_metric = 1e6
        for epoch in range(self.epochs):
            # 训练
            train_loss = self.train_one_epoch(epoch, train_loader, criterion, optimizer)
            # 验证
            val_data, val_loss = self.validation_one_epoch(epoch, val_loader, criterion)
            # 更新学习率
            lr_scheduler.step(val_loss[0])

            # 日志记录
            self.writer.add_scalars("epoch_loss", dict(train_loss=train_loss[0], val_loss=val_loss[0]),
                               global_step=epoch)
            self.writer.add_scalar("epoch_lr", optimizer.param_groups[0]['lr'], global_step=epoch)

            # 记录模型
            if not os.path.exists(self.ckpt_dir):
                os.makedirs(self.ckpt_dir)

            if best_metric > val_loss[0]:
                best_metric = val_loss[0]
                # 记录最佳模型
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, "best.pth"))
                # 记录实验数据
                with torch.no_grad():
                    inputs, labels, outputs = val_data
                    inputs = inputs.cpu()[:1, ...]                  # B, N, 9
                    labels = labels.cpu()[:1, ...]                  # B, N
                    outputs = outputs.cpu()[:1, ...]                # B, N, num_classes
                    pred_labels = torch.argmax(outputs, dim=-1)     # B, N
                    self.writer.add_mesh("points", inputs[:, :, :3], colors=(inputs[:, :, 3:6] * 255).to(torch.uint8), global_step=epoch)
                    self.writer.add_mesh("labels", inputs[:, :, :3], colors=colors[labels], global_step=epoch)
                    self.writer.add_mesh("pred_labels", inputs[:, :, :3], colors=colors[pred_labels], global_step=epoch)

            torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, "last.pth"))

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--data_dir", type=str, default="../datasets/chicken3d", help="dataset load directory")
    parse.add_argument("--ckpt_dir", type=str, default="../ckpts/pointnet2", help="model checkpoint save directory")
    parse.add_argument("--ckpt_file", type=str, default=None, help="model pretrained checkpoint file")
    parse.add_argument("--log_dir", type=str, default="./runs/pointnet2/run_01", help="model train log(tensorboard) save directory")
    parse.add_argument("--num_classes", type=int, default=2, help="model classes number")
    parse.add_argument("--batch_size", type=int, default=64, help="model train batch size")
    parse.add_argument("--lr", type=float, default=1e-3, help="model train learning rate")
    parse.add_argument("--epochs", type=int, default=100, help="model train total epoch number")
    parse.add_argument("--device", type=int, default=0, help="model run device, default=0")

    args = parse.parse_args()

    trainer = PCDModelTrainer(
        model_name="PointNet2",
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir,
        ckpt_file=args.ckpt_file,
        log_dir=args.log_dir,
        in_point_dim=3,
        in_feat_dim=3,
        sampled_point_num=2048,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        device=args.device,
        lr=args.lr,
        epochs=args.epochs)
    trainer.run()
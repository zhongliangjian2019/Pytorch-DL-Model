"""
@brief: 分割模型训练脚本
"""
import torch
from torch import optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from dataset import SegmentDataset as MyDataset
from model_loss import SegmentLoss
from model_zoo import UNet, PPLiteSeg
import os
import logging
import datetime
import shutil
import argparse

# 日志文件格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class TrainConfigure:
    """分割模型配置项"""
    def __init__(self):
        self.data_dir = ''
        self.batch_size = 256
        self.num_classes = 2
        self.in_channel = 1
        self.image_size = (640, 640)
        self.model = 'unet'
        self.learning_rate = 1e-3
        self.epochs = 300
        self.class_weight = [1, 1, 1]
        self.load_num_workers = 16
        self.is_wandb = True
        self.checkpoint = None
        self.device = 'cuda:0'

class ModelTrainer:
    """分割模型训练器"""
    def __init__(self, args):
        self.logger = logging.getLogger("model_trainer")
        self.config = self.get_train_config(args)

    def get_train_config(self, args):
        cfg = TrainConfigure()
        cfg.data_dir = args.data_dir
        cfg.num_classes = args.num_classes
        cfg.batch_size = args.batch_size
        cfg.learning_rate = args.learn_rate
        cfg.in_channel = args.in_channels
        cfg.model = args.model
        cfg.checkpoint = args.checkpoint
        cfg.epochs = args.epochs
        cfg.load_num_workers = 4
        cfg.is_wandb = True

        self.logger.info("segment model training configure:")
        self.logger.info("data_dir: {0}".format(args.data_dir))
        self.logger.info("in_channels: {0}".format(args.in_channels))
        self.logger.info("num_classes: {0}".format(args.num_classes))
        self.logger.info("batch_size: {0}".format(args.batch_size))
        self.logger.info("learn_rate: {0}".format(args.learn_rate))
        self.logger.info("iter_epochs: {0}".format(args.epochs))
        self.logger.info("checkpoint: {0}".format(args.checkpoint))
        return cfg

    def get_dataset_loader(self):
        """
        功能描述：获取数据加载器
        return: train_loader, n_train, val_loader, n_val
        """
        assert os.path.exists(self.config.data_dir), "find not data directory"

        # 1.创建数据集
        train_set = MyDataset(self.config.data_dir, mode='train', in_channel=self.config.in_channel,
                                num_classes=self.config.num_classes, image_size=self.config.image_size)
        val_set = MyDataset(self.config.data_dir, mode='val', in_channel=self.config.in_channel,
                              num_classes=self.config.num_classes, image_size=self.config.image_size)

        # 2.划分训练集与验证集
        n_train = len(train_set)
        n_val   = len(val_set)

        # 3.创建数据加载器
        loader_args = dict(batch_size=self.config.batch_size, num_workers=self.config.load_num_workers, pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, drop_last=False,  **loader_args)
        val_loader   = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

        return train_loader, n_train, val_loader, n_val

    def config_model(self):
        """
        功能描述：配置模型（训练设备、损失函数、优化器、学习率及衰减机制）
        """
        # 1.创建模型
        if self.config.model == "pp_liteseg":
            model = PPLiteSeg(num_class=self.config.num_classes, n_channel=self.config.in_channel,
                              encoder_channels=(16, 32, 64, 128, 256),
                              encoder_type='stdc3', fusion_type='both')
        else:
            model = UNet(n_channels=self.config.in_channel, n_classes=self.config.num_classes, bilinear=True,
                         is_eval=False)

        # 2.加载预训练参数
        if self.config.checkpoint is not None:
            # 获取当前模型的 state_dict
            model_dict = model.state_dict()
            # 加载预训练权重
            pretrained_dict = torch.load(self.config.checkpoint, weights_only=True)
            # 过滤预训练权重：只加载键值匹配且形状相同的权重
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                if k in model_dict and v.size() == model_dict[k].size()}
            # 更新当前模型的 state_dict
            model_dict.update(pretrained_dict)
            # 将更新后的 state_dict 加载到模型中
            model.load_state_dict(model_dict)

        # 1.配置训练设备
        device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        model.to(device=device)
        logging.info("train device: {0}".format(device))

        # 2.损失函数
        criterion = SegmentLoss()

        # 3.优化器
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=1e-2)

        # 4.学习率监督器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, min_lr=1e-5)

        return model, device, criterion, optimizer, scheduler

    def train_one_epoch(self, model, train_loader, device, criterion, optimizer, epoch, epochs):
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

        train_loss = 0
        try:
            with tqdm(desc="train epoch[{0}/{1}]".format(epoch, epochs), total=len(train_loader)) as pbar:
                for data in train_loader:
                    # 加载数据
                    images, masks = data['image'], data['mask']

                    assert images.size(1) == model.in_channel, "train error: input channel and network input channel mismatching"

                    images = images.to(device=device)
                    masks = masks.to(device='cpu')

                    # 前向推理
                    predictions = model(images)
                    predictions = predictions.to(device='cpu')
                    # 计算损失
                    loss = criterion(predictions, masks)
                    train_loss += loss.item()
                    # 梯度清零
                    optimizer.zero_grad(set_to_none=True)
                    # 反向传播
                    loss.backward()
                    # 更新参数
                    optimizer.step()

                    pbar.set_postfix(loss="{0:.4f}".format(loss.item()))
                    pbar.update()

                if len(train_loader) != 0:
                    train_loss /= len(train_loader)

                pbar.set_postfix(train_loss=train_loss, lr=optimizer.param_groups[0]['lr'])
        except KeyboardInterrupt:
            pbar.close()

        return images, masks, predictions, train_loss

    def validation(self, model, device, val_loader, epoch, epochs, criterion):
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
            try:
                with tqdm(desc="valid epoch[{0}/{1}]".format(epoch, epochs), total=len(val_loader)) as pbar:
                    for data in val_loader:
                        # 加载数据
                        images, masks = data['image'], data['mask']


                        assert images.size(1) == model.in_channel, "valid error: input channel and network input channel mismatching"

                        images = images.to(device=device)
                        masks = masks.to(device='cpu')

                        # 前向推理
                        predictions = model(images)
                        predictions = predictions.to(device='cpu')

                        # 计算损失
                        loss = criterion(predictions, masks)
                        total_loss += loss.item()

                        pbar.set_postfix(score="{0:.4f}".format(loss.item()))
                        pbar.update()

                    total_loss /= len(val_loader)

                    pbar.set_postfix(val_loss="{0:.4f}".format(total_loss))

            except KeyboardInterrupt:
                pbar.close()

        return images, masks, predictions, total_loss

    def run(self):
        """功能描述：模型训练"""
        # 3.创建模型保存节点目录
        cur_time = datetime.datetime.now()
        ckpt_dir = "../ckpts/ckpt_{0}".format(cur_time.strftime("%Y%m%d-%H%M%S"))
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        os.makedirs(ckpt_dir)

        # 4.获取数据加载器
        train_loader, n_train, val_loader, n_val = self.get_dataset_loader()

        # 5.配置训练模型
        model, device, criterion, optimizer, scheduler = self.config_model()

        # 6.初始化训练监测
        if self.config.is_wandb:
            wandb_run = wandb.init(dir='../train_log', project='segment', entity='zhongliangjian',
                                    name=cur_time.strftime("%Y%m%d-%H%M%S"))
            wandb_run.config.update(dict(epochs=self.config.epochs, batch_size=self.config.batch_size,
                                          learning_rate=self.config.learning_rate))

        # 5.开始训练（训练 + 验证）
        best_val_loss = 1e6
        epochs = self.config.epochs
        for epoch in range(1, self.config.epochs):
            self.logger.info("epoch: [{0} / {1}]".format(epoch, epochs))
            # 训练
            images, truths, preds, train_loss = self.train_one_epoch(model, train_loader, device,
                                                                           criterion, optimizer, epoch, epochs)
            # 验证
            images, truths, preds, val_loss = self.validation(model, device, val_loader, epoch,
                                                                      epochs, criterion)
            # 学习率监督
            scheduler.step(val_loss)
            # 过程日志
            with torch.no_grad():
                # 保存最新模型
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "last.pth"))

                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pth"))

                # 监控信息
                if self.config.is_wandb:
                    wandb.log({'learning rate': optimizer.param_groups[0]['lr'],
                                'train_loss': train_loss,
                                'val_loss': best_val_loss,
                                'epoch': epoch})

if __name__ == "__main__":
    """模块测试"""
    parser = argparse.ArgumentParser(description="segment model training script")
    parser.add_argument('-b', '--batch_size', type=int, default=16, help="setting batch size, default = 16")
    parser.add_argument('-lr', '--learn_rate', type=float, default=1e-3, help="setting initial learning rate, default = 0.001")
    parser.add_argument('-e', '--epochs', type=int, default=300, help="setting training total epochs number, default = 300")
    parser.add_argument('-d', '--data_dir', type=str, default=None, help="setting training data load directory")
    parser.add_argument('-c', '--num_classes', type=int, default=1, help="setting model classify number, should contain background")
    parser.add_argument('-ch', '--in_channels', type=int, default=3, help="setting model input channel number, default = 3")
    parser.add_argument('-m', '--model', type=str, default='unet', help="setting model mode, available model set {unet, ppliteseg}")
    parser.add_argument('-ckp', '--checkpoint', type=str, default=None, help="setting model load checkpoint")
    args = parser.parse_args()
    trainer = ModelTrainer(args)
    trainer.run()

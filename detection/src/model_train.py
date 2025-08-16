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
from model_dnet import DetUNet
from loss_func import DetectLoss
import os
import logging
import datetime
import shutil
import argparse
from model_inference import decode_bbox
from wandb_show import wandb_boxes2d
import os
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
# 日志文件格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class TrainConfigure:
    """分割模型配置项"""
    def __init__(self):
        self.data_dir = ''
        self.batch_size = 256
        self.num_classes = 1
        self.in_channel = 1
        self.image_size = (512, 512)
        self.model = 'unet'
        self.learning_rate = 5e-4
        self.epochs = 500
        self.class_weight = [1, 1, 1]
        self.load_num_workers = 4
        self.is_wandb = True
        self.device = "cuda:0"
        self.checkpoint = None

class ModelTrainer:
    """模型训练器"""
    def __init__(self, args):
        self.logger = logging.getLogger("model_trainer")
        self.config = self.get_train_config(args)

    def get_train_config(self, args):
        """获取模型训练配置"""
        cfg = TrainConfigure()
        cfg.data_dir = args.data_dir
        cfg.batch_size = args.batch_size
        cfg.lr = args.learn_rate
        cfg.in_channel = args.in_channels
        cfg.checkpoint = args.checkpoint
        cfg.epochs = args.epochs

        logging.info("model training configure:")
        logging.info("  data_dir:       {0}".format(cfg.data_dir))
        logging.info("  in_channels:    {0}".format(cfg.in_channel))
        logging.info("  num_classes:    {0}".format(cfg.num_classes))
        logging.info("  batch_size:     {0}".format(cfg.batch_size))
        logging.info("  learn_rate:     {0}".format(cfg.lr))
        logging.info("  iter_epochs:    {0}".format(cfg.epochs))
        logging.info("  checkpoint:     {0}".format(cfg.checkpoint))

        return cfg

    def get_dataset_loader(self):
        """
        功能描述：获取数据加载器
        return: train_loader, val_loader
        """
        assert os.path.exists(self.config.data_dir), "find not data directory"

        # 1.创建数据集
        train_set = MyDataSet(data_dir=self.config.data_dir,
                              mode='train',
                              in_channel=self.config.in_channel,
                              num_classes=self.config.num_classes,
                              image_size=self.config.image_size)

        val_set = MyDataSet(data_dir=self.config.data_dir,
                            mode='val',
                            in_channel=self.config.in_channel,
                            num_classes=self.config.num_classes,
                            image_size=self.config.image_size)

        # 3.创建数据加载器
        loader_args = dict(batch_size=self.config.batch_size, num_workers=self.config.load_num_workers, pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

        return train_loader, val_loader

    def config_model(self):
        """
        功能描述：配置模型（训练设备、损失函数、优化器、学习率及衰减机制）
        """
        # 1.创建模型
        model = DetUNet(in_channel=self.config.in_channel, num_classes=self.config.num_classes)

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

        # 3.配置训练设备
        device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        model.to(device=device)
        logging.info("train device: {0}".format(device))

        # 4.损失函数
        criterion = DetectLoss()

        # 5.优化器
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

        # 6.学习率监督器
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
        total_loss = 0
        total_c_loss = 0
        total_r_loss = 0
        try:
            with tqdm(desc="train epoch[{0}/{1}]".format(epoch, epochs), total=len(train_loader)) as pbar:
                for data in train_loader:
                    # 加载数据
                    batch_images, batch_truth = data
                    batch_images = batch_images.to(device)

                    hm, wh, offset = model(batch_images)
                    hm = hm.cpu()
                    wh = wh.cpu()
                    offset = offset.cpu()
                    num_classes = model.num_classes
                    batch_hms = batch_truth[:, :, :, 0:num_classes]
                    batch_whs = batch_truth[:, :, :, num_classes: num_classes + 2]
                    batch_offsets = batch_truth[:, :, :, num_classes + 2: num_classes + 4]
                    batch_reg_masks = batch_truth[:, :, :, -1]
                    c_loss = criterion.focal_loss(hm, batch_hms)
                    wh_loss = 0.1 * criterion.reg_l1_loss(hm, wh, batch_whs, batch_reg_masks)
                    off_loss = criterion.reg_l1_loss(hm, offset, batch_offsets, batch_reg_masks)

                    loss = c_loss + wh_loss + off_loss

                    total_loss += loss.item()
                    total_c_loss += c_loss.item()
                    total_r_loss += wh_loss.item() + off_loss.item()

                    # 梯度清零
                    optimizer.zero_grad(set_to_none=True)
                    # 反向传播
                    loss.backward()
                    # 更新参数
                    optimizer.step()

                    pbar.set_postfix(all_loss="{0:.4f}".format(loss.item()),
                                     cla_loss="{0:.4f}".format(c_loss.item()),
                                     reg_loss="{0:.4f}".format(wh_loss.item() + off_loss.item()))
                    pbar.update()

                if len(train_loader) != 0:
                    total_loss /= len(train_loader)
                    total_c_loss /= len(train_loader)
                    total_r_loss /= len(train_loader)

                pbar.set_postfix(all_loss=total_loss, cla_loss=total_c_loss, reg_loss=total_r_loss,
                                 lr=optimizer.param_groups[0]['lr'])
        except KeyboardInterrupt:
            pbar.close()

        return batch_images, hm, wh, offset, total_loss

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
            total_c_loss = 0
            total_r_loss = 0
            try:
                with tqdm(desc="valid epoch[{0}/{1}]".format(epoch, epochs), total=len(val_loader)) as pbar:
                    for data in val_loader:
                        # 加载数据
                        batch_images, batch_truth = data

                        hm, wh, offset = model(batch_images.to(device))
                        hm = hm.cpu()
                        wh = wh.cpu()
                        offset = offset.cpu()
                        num_classes = model.num_classes
                        batch_hms = batch_truth[:, :, :, 0:num_classes]
                        batch_whs = batch_truth[:, :, :, num_classes: num_classes + 2]
                        batch_offsets = batch_truth[:, :, :, num_classes + 2: num_classes + 4]
                        batch_reg_masks = batch_truth[:, :, :, -1]
                        c_loss = criterion.focal_loss(hm, batch_hms)
                        wh_loss = 0.1 * criterion.reg_l1_loss(hm, wh, batch_whs, batch_reg_masks)
                        off_loss = criterion.reg_l1_loss(hm, offset, batch_offsets, batch_reg_masks)

                        loss = c_loss + wh_loss + off_loss

                        total_loss += loss.item()
                        total_c_loss += c_loss.item()
                        total_r_loss += wh_loss.item() + off_loss.item()

                        pbar.set_postfix(score="{0:.4f}".format(loss.item()))
                        pbar.update()

                    if len(val_loader) != 0:
                        total_loss /= len(val_loader)
                        total_c_loss /= len(val_loader)
                        total_r_loss /= len(val_loader)

                    pbar.set_postfix(all_loss=total_loss, cla_loss=total_c_loss, reg_loss=total_r_loss)

            except KeyboardInterrupt:
                pbar.close()

        return batch_images, batch_hms, batch_whs, batch_offsets, hm, wh, offset, total_loss

    def run(self):
        """功能描述：模型训练"""
        # 1.创建模型保存节点目录
        cur_time = datetime.datetime.now()
        ckpt_dir = "../ckpts/ckpt_{0}".format(cur_time.strftime("%Y%m%d-%H%M%S"))
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        os.makedirs(ckpt_dir)

        # 2.获取数据加载器
        train_loader, val_loader = self.get_dataset_loader()

        # 3.配置训练模型
        model, device, criterion, optimizer, scheduler = self.config_model()

        # 4.初始化训练监测
        if self.config.is_wandb:
            wandb_run = wandb.init(dir='../train_log', project='detection', entity='zhongliangjian',
                                    name=cur_time.strftime("%Y%m%d-%H%M%S"))
            wandb_run.config.update(dict(epochs=self.config.epochs, batch_size=self.config.batch_size,
                                          learning_rate=self.config.learning_rate))

        # 5.开始训练（训练 + 验证）
        best_val_loss = 1e6
        for epoch in range(1, self.config.epochs):
            self.logger.info("epoch: [{0} / {1}]".format(epoch, self.config.epochs))

            # 训练
            batch_images, hm, wh, offset, train_loss = \
                self.train_one_epoch(model, train_loader, device, criterion, optimizer, epoch, self.config.epochs)

            # 验证
            batch_images, batch_hms, batch_whs, batch_regs, hm, wh, offset, val_loss = \
                self.validation(model, device, val_loader, epoch, self.config.epochs, criterion)

            # 学习率监督
            scheduler.step(val_loss)

            with torch.no_grad():
                # 保存训练节点
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "last.pth"))

                # 保存得分最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pth"))

                    # 精度最佳模型监测
                    if self.config.is_wandb:
                        with torch.no_grad():
                            p_hms = hm.cpu().numpy()
                            p_whs = wh.cpu().numpy()
                            p_offsets = offset.cpu().numpy()
                            p_bboxes = decode_bbox(p_hms, p_whs, p_offsets, 0.5)
                            p_bboxes = p_bboxes[0]
                            g_hms = batch_hms.cpu().numpy().transpose([0, 3, 1, 2])
                            g_whs = batch_whs.cpu().numpy().transpose([0, 3, 1, 2])
                            g_offsets = batch_regs.cpu().numpy().transpose([0, 3, 1, 2])
                            g_bboxes = decode_bbox(g_hms, g_whs, g_offsets, 0.5)
                            g_bboxes = g_bboxes[0]
                            show_image = batch_images.cpu().numpy()
                            show_image = np.squeeze(show_image[0])
                            img = wandb_boxes2d(show_image, p_bboxes, g_bboxes, class_labels={0: "d"})
                            wandb.log({"droplet_detection": img})
                            show_heatmap = p_hms[0, 0] * 255
                            show_heatmap = show_heatmap.astype(np.uint8)
                            show_heatmap = cv2.applyColorMap(show_heatmap, cv2.COLORMAP_JET)
                            wandb.log({"heatmap": wandb.Image(show_heatmap)})

                # 监控信息
                if self.config.is_wandb:
                    wandb.log({'learning rate': optimizer.param_groups[0]['lr'],
                                'train_loss': train_loss,
                                'val_loss': best_val_loss,
                                'epoch': epoch})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="segment model training script")
    parser.add_argument('-b', '--batch_size', type=int, default=8, help="setting batch size, default = 16")
    parser.add_argument('-lr', '--learn_rate', type=float, default=1e-3, help="setting initial learning rate, default = 0.001")
    parser.add_argument('-e', '--epochs', type=int, default=10, help="setting training total epochs number, default = 1000")
    parser.add_argument('-d', '--data_dir', type=str, default=r'..\data', help="setting training data load directory")
    parser.add_argument('-c', '--num_classes', type=int, default=1, help="setting model classify number, should contain background")
    parser.add_argument('-ch', '--in_channels', type=int, default=3, help="setting model input channel number")
    parser.add_argument('-ckp', '--checkpoint', type=str, default=None, help="setting model load checkpoint")

    args = parser.parse_args()

    trainer = ModelTrainer(args)

    trainer.run()


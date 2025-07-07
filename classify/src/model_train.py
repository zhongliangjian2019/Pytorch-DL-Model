"""模型训练"""
import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as tf
import wandb
import argparse
from tqdm import tqdm
import os
import logging
import datetime
import shutil
from dataset import ClassDataset as MyDataset
from model_loss import ClassLoss
from model_zoo import MobileNetV3_Small

# 日志文件格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class argConfig:
    """模型配置项"""
    def __init__(self):
        self.data_dir = ''
        self.batch_size = 256
        self.num_classes = 3
        self.in_channel = 3
        self.image_size = 224
        self.model = 'unet'
        self.learning_rate = 1e-3
        self.epochs = 1000
        self.class_weight = [1, 1, 1]
        self.load_num_workers = 16
        self.is_wandb = True
        self.checkpoint = None

class ModelTrainer:
    def __init__(self, args):
        self.logger = logging.getLogger("class trainer")
        self.train_config = self.get_train_args(args)

    def get_train_args(self, args):
        self.logger.info("classify model training configure:")
        self.logger.info("data_dir: {0}".format(args.data_dir))
        self.logger.info("in_channels: {0}".format(args.in_channels))
        self.logger.info("num_classes: {0}".format(args.num_classes))
        self.logger.info("batch_size: {0}".format(args.batch_size))
        self.logger.info("learn_rate: {0}".format(args.learn_rate))
        self.logger.info("iter_epochs: {0}".format(args.epochs))
        self.logger.info("checkpoint: {0}".format(args.checkpoint))

        cfg = argConfig()
        cfg.data_dir = args.data_dir
        cfg.batch_size = args.batch_size
        cfg.lr = args.learn_rate
        cfg.in_channel = args.in_channels
        cfg.model = args.model
        cfg.checkpoint = args.checkpoint
        cfg.epochs = args.epochs
        cfg.is_wandb = True
        cfg.load_num_workers = 16
        return cfg

    def get_data_loader(self):
        """
        功能描述：获取数据加载器
        param: data_dir: 数据目录
        return: train_loader, n_train, val_loader, n_val
        """
        assert os.path.exists(self.train_config.data_dir), "find not data directory"

        # 1.创建数据集
        train_set = MyDataset(os.path.join(self.train_config.data_dir, 'train.txt'), mode='train',
                              in_channel=self.train_config.in_channel, in_size=self.train_config.image_size)
        val_set   = MyDataset(os.path.join(self.train_config.data_dir, 'val.txt'), mode='val',
                              in_channel=self.train_config.in_channel, in_size=self.train_config.image_size)

        # 2.划分训练集与验证集
        n_train = len(train_set)
        n_val   = len(val_set)

        # 3.创建数据加载器
        loader_args = dict(batch_size=self.train_config.batch_size, num_workers=self.train_config.load_num_workers,
                           pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
        val_loader   = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

        return train_loader, n_train, val_loader, n_val

    def config_model(self):
        """
        功能描述：配置模型（训练设备、损失函数、优化器、学习率及衰减机制）
        """
        # 0.创建模型
        model = MobileNetV3_Small(in_channel=self.train_config.in_channel, num_classes=self.train_config.num_classes)

        if self.train_config.checkpoint is not None:
            model_dict = model.state_dict()
            pretrained_dict = torch.load(self.train_config.checkpoint, map_location='cpu')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)

        # 1.配置训练设备
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model.to(device=device)
        logging.info("train device: {0}".format(device))

        # 2.损失函数
        if self.train_config.class_weight is None:
            weight = torch.tensor([1 for i in range(model.n_classes)], dtype=torch.float32)
        else:
            weight = torch.tensor(self.train_config.class_weight, dtype=torch.float32)
        criterion = ClassLoss(weight)

        # 3.优化器
        optimizer = optim.Adam(model.parameters(), lr=self.train_config.learning_rate, weight_decay=1e-6)

        # 4.学习率监督器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100)

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
                    images, labels = data['image'], data['label']

                    assert images.shape[1] == model.in_channel, \
                        "train error: input channel and network input channel mismatching"

                    images = images.to(device=device, dtype=torch.float32)
                    labels = labels.to(device='cpu', dtype=torch.int64)

                    # 前向推理
                    pred_labels = model(images)
                    pred_labels = pred_labels.to(device='cpu')
                    # 计算损失
                    loss = criterion(pred_labels, labels)
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

                pbar.set_postfix(train_loss=train_loss)
        except KeyboardInterrupt:
            pbar.close()

        return images, labels, pred_labels, train_loss

    def validation(self, model, device, val_loader, criterion, epoch, epochs):
        """
        功能描述：模型验证
        :param model: 网络模型
        :param device: 训练设备
        :param val_loader: 验证集加载器
        :param criterion: 损失函数
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
            val_loss = 0
            correct_rate = 0
            try:
                with tqdm(desc="valid epoch[{0}/{1}]".format(epoch, epochs), total=len(val_loader)) as pbar:
                    for data in val_loader:
                        # 加载数据
                        images, labels = data['image'], data['label']
                        assert images.shape[1] == model.in_channel, \
                            "valid error: input channel and network input channel mismatching"

                        images = images.to(device=device, dtype=torch.float32)
                        labels = labels.to(device='cpu', dtype=torch.int64)

                        # 前向推理
                        pred_labels = model(images)
                        pred_labels = pred_labels.to(device='cpu')

                        # 计算损失
                        loss = criterion(pred_labels, labels)
                        val_loss += loss.item()
                        # 计算正确率
                        correct_rate += (torch.argmax(tf.softmax(pred_labels, dim=1), dim=1) == labels).sum().item()

                        pbar.set_postfix(loss="{0:.4f}".format(loss.item()))
                        pbar.update()

                    val_loss /= len(val_loader)
                    correct_rate /= len(val_loader.dataset)

                    pbar.set_postfix(val_loss=val_loss, correct_rate=correct_rate)

            except KeyboardInterrupt:
                pbar.close()

        return images, labels, pred_labels, val_loss, correct_rate

    def run(self):
        """
        功能描述：模型训练
        :param model: 网络模型
        :param data_dir: 数据目录
        :param batch_size: 批处理大小
        :param learning_rate: 初始学习率
        :param epochs: 总的迭代轮次
        :return: None
        """
        # 创建模型保存节点目录
        cur_time = datetime.datetime.now()
        ckpt_dir = "../ckpts/ckpt_{0}".format(cur_time.strftime("%Y%m%d-%H%M%S"))
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        os.makedirs(ckpt_dir)

        # 1.获取数据加载器
        train_loader, n_train, val_loader, n_val = self.get_data_loader()

        # 2.配置训练模型
        model, device, criterion, optimizer, scheduler = self.config_model()

        # 3.初始化训练监测
        if self.train_config.is_wandb:
            experiment = wandb.init(project='classify', entity='zhongliangjian', name=cur_time.strftime("%Y%m%d-%H%M%S"))
            experiment.config.update(dict(epochs=self.train_config.epochs, batch_size=self.train_config.batch_size,
                                          learning_rate=self.train_config.learning_rate))

        # 5.开始训练（训练 + 验证）
        best_correct_rate = 0
        epochs = self.train_config.epochs
        for epoch in range(epochs):
            logging.info("epoch: [{0} / {1}]".format(epoch, epochs))

            # 训练
            images, labels, p_labels, train_loss = self.train_one_epoch(model, train_loader, device,
                                                                        criterion, optimizer, epoch, epochs)

            # 验证
            images, labels, p_labels, val_loss, correct_rate = self.validation(model, device, val_loader,
                                                                               criterion, epoch, epochs)

            # 学习率监督
            scheduler.step(val_loss)

            with torch.no_grad():
                # 保存训练节点
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "last.pth"))

                # 保存精度最佳模型
                if correct_rate > best_correct_rate:
                    best_correct_rate = correct_rate
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pth"))
                    # 精度最佳模型监测
                    if self.train_config.is_wandb:
                        wandb_images = []
                        for i in range(min(10, labels.shape[0])):
                            pred_id = torch.argmax(tf.softmax(p_labels, dim=1), dim=1)[i].item()
                            pred_score = tf.softmax(p_labels, dim=1)[i, pred_id].item()
                            wandb_images.append(wandb.Image(images[i].cpu().numpy().transpose((2, 1, 0)),
                                caption="image_%2d: label-%d, pred-%d, score-%.4f" % (i, labels[i].item(),
                                            pred_id, pred_score)))

                        wandb.log({'labels_vs_predicts': wandb_images})

                # 监控信息
                if self.train_config.is_wandb:
                    wandb.log({'learning rate': optimizer.param_groups[0]['lr'],
                                    'train_loss': train_loss,
                                    'val_loss': val_loss,
                                    'val_acc': correct_rate,
                                    'epoch': epoch})


if __name__ == "__main__":
    """模型训练"""
    parser = argparse.ArgumentParser(description="segment model training script")
    parser.add_argument('-b', '--batch_size', type=int, default=256, help="setting batch size, default = 16")
    parser.add_argument('-lr', '--learn_rate', type=float, default=1e-3, help="setting initial learning rate, default = 0.001")
    parser.add_argument('-e', '--epochs', type=int, default=1000, help="setting training total epochs number, default = 1000")
    parser.add_argument('-d', '--data_dir', type=str, default=r'../../DetectionModel/dataset/tube_data_20240923', help="setting training data load directory")
    parser.add_argument('-c', '--num_classes', type=int, default=3, help="setting model classify number, should contain background")
    parser.add_argument('-ch', '--in_channels', type=int, default=3, help="setting model input channel number")
    parser.add_argument('-m', '--model', type=str, default='unet', help="setting model mode, 0 - dawei, 1 - maccura")
    parser.add_argument('-ckp', '--checkpoint', type=str, default='../checkpoints/tube_detection_best.pth', help="setting model load checkpoint")
    args = parser.parse_args()
    trainer = ModelTrainer(args)
    trainer.run()

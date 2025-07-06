"""
模型训练损失函数
"""
import torch.nn as nn
from torch.nn import functional as F
import torch

def GetDiceScore(logits, targets):
    """计算Dice得分"""
    batch_num = targets.size(0)
    probs = torch.sigmoid(logits)
    targets = F.one_hot(targets.cpu().long(), num_classes=logits.size(1)).permute(0, 3, 1, 2)
    m1 = probs.reshape(batch_num, -1)
    m2 = targets.reshape(batch_num, -1)

    # 计算交集与并集
    inter_set = (m1 * m2).sum(1)
    union_set = m1.sum(1) + m2.sum(1)

    dice_score = 2 * (inter_set + 1) / (union_set + 1)  # +1避免极端情况
    dice_score = dice_score.sum() / batch_num

    return dice_score

class SoftDiceLoss(nn.Module):
    def __init__(self, alpha: float = 3):
        super(SoftDiceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, logists, targets):
        """
        Soft Dice Loss
        :param logits: 模型预测 [B, C, H ,W]
        :param targets: 标签 [B, H, W]
        :return: loss
        """
        batch_num = logists.size(0)
        channel_num = logists.size(1)
        probs = torch.sigmoid(logists)
        total_loss = 0
        for ch in range(0, channel_num):
            m1 = probs[:, ch, :, :].reshape(batch_num, -1)
            m2 = targets[:, ch, :, :].reshape(batch_num, -1)

            # 计算交集与并集
            inter_set = (m1 * m2).sum(1)
            union_set = m1.sum(1) + m2.sum(1)

            score = 2 * (inter_set + 1) / (union_set + 1)  # +1避免极端情况
            loss = 1 - score.sum() / batch_num
            total_loss += loss

        dice_loss = total_loss / channel_num

        return dice_loss

class BCELoss2d(nn.Module):
    def __init__(self, weight=None):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, reduction='mean')

    def forward(self, logists, targets):
        targets = F.one_hot(targets.cpu().long(), num_classes=logists.size(1)).permute(0, 3, 1, 2)
        probs = torch.sigmoid(logists)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss 实现
        :param alpha: 类别权重，用于平衡正负样本（默认 0.25）
        :param gamma: 调节因子，用于减少易分类样本的权重（默认 2.0）
        :param reduction: 损失计算方式，可选 'mean'、'sum' 或 'none'（默认 'mean'）
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        计算 Focal Loss
        :param inputs: 模型输出的 logits（未经过 sigmoid/softmax）
        :param targets: 真实标签（0 或 1）
        :return: Focal Loss 值
        """
        # 对输入进行 sigmoid 激活
        p = torch.sigmoid(inputs)

        # 计算交叉熵损失
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')

        # 计算 p_t
        p_t = p * targets + (1 - p) * (1 - targets)

        # 计算 Focal Loss
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        # 应用 alpha 权重
        if self.alpha >= 0:
            alpha_t = self.alpha * (1 - targets) + (1 - self.alpha) * targets
            loss = alpha_t * loss

        # 根据 reduction 参数返回损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class SegmentLoss(nn.Module):
    """分割模型损失函数"""
    def __init__(self):
        super(SegmentLoss, self).__init__()
        self.dice_loss = SoftDiceLoss()
        self.focal_loss = FocalLoss()

    def forward(self, logits, targets):
        if logits.size(1) > 1:
            targets = F.one_hot(targets.cpu().long(), num_classes=logits.size(1)).permute(0, 3, 1, 2)
        else:
            targets = torch.unsqueeze(targets, dim=1)
        loss = self.dice_loss(logits, targets) + 10 * self.focal_loss(logits, targets)
        return loss

if __name__ == "__main__":
    """模块测试"""
    input = torch.randn(size=(1, 1, 640, 640))
    target = torch.randint(0, 1, size=(1, 1, 640, 640))
    focal_loss = FocalLoss()
    loss = focal_loss(input, target)
    print(loss.item())
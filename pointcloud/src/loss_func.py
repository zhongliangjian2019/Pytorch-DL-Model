"""
@brief 损失函数
    - 分类损失 cross_entropy / focal_loss
    - 特征矩阵损失
"""
import torch
from torch import nn, dtype
from torch.nn import functional as F

class LossFunc(nn.Module):
    """损失函数"""
    def __init__(self, alpha: float|list = 1.0, gamma: float = 2.0,
                 trans_loss_scale: float = 0.001):
        """
        :param alpha: focal_loss 类别权重
        :param gamma: focal_loss 难易样本权重
        :param trans_loss_scale: 变换矩阵损失权重
        """
        super().__init__()
        self.scale = trans_loss_scale
        self.trans_loss = TransformLoss()
        self.class_loss = FocalLoss(alpha, gamma)
        self.dice_loss = DiceLoss(alpha)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, pred_trans: torch.Tensor = None):
        """
        :param pred: 分类-(B, C) / 分割-(B, N, C)
        :param target: 分类-(B,) / 分割-(B, N)
        :param pred_trans: (B, D1, D1)
        :return: tuple(loss-总损失, c_loss-分类损失, t_loss-变换损失)
        """
        c_loss = self.class_loss(pred, target)
        d_loss = self.dice_loss(pred, target)
        if pred_trans is not None:
            t_loss = self.trans_loss(pred_trans) * self.scale
            loss = c_loss + t_loss + d_loss
        else:
            loss = c_loss + d_loss
            t_loss = d_loss
        return (loss, c_loss, t_loss)

class TransformLoss(nn.Module):
    """变换损失"""
    def __init__(self):
        super().__init__()

    def forward(self, trans: torch.Tensor):
        """
        计算 Loss = ||I - A * A_t||F2
        :param trans: (B, dims, dims)
        :return:
        """
        A = trans
        I = torch.eye(A.size(1), device=trans.device)[None, :, :]   # (1, dims, dims)
        A_t = A.transpose(2, 1)
        loss = torch.norm(I - torch.bmm(A, A_t), dim=(1, 2))
        loss = torch.mean(loss)
        return loss

class FocalLoss(nn.Module):
    """分类损失"""
    def __init__(self, alpha: float|list = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        :param pred: 模型预测类别概率, 经过sigmoid(二分类) / softmax(多分类)的输出, 分类-(B, C) / 分割-(B, N, C)
        :param target: 未进行one-hot编码的标签, 分类-(B,) / 分割-(B, N)
        :return:
        """
        # B: batch_size, C: num_classes
        B, C = pred.size(0), pred.size(-1)
        # 模型预测概率
        probs = pred.reshape(-1, C)    # (*, C)
        # 标签one-hot编码: 多分类处理, 二分类无需编码
        if C > 1:
            # 多分类
            targets = F.one_hot(target, num_classes=C).reshape(-1, C).float()
            pt = probs.clip(min=1e-8, max=1-1e-8)
        else:
            # 二分类->转换为多分类处理
            target = target.reshape(-1, 1)  # (*, 1)
            targets = torch.cat([target, 1 - target], dim=-1).float()  # (*, 2)
            pt = torch.cat([probs, 1 - probs], dim=-1)  # (*, 2)
        # 模型预测类别概率 - pt
        pt = pt.clip(min=1e-8, max=1-1e-8)
        # 类别权重 - αt
        if isinstance(self.alpha, float):
            if C == 1:
                # 二分类
                alpha = torch.tensor([self.alpha, 1 - self.alpha], device=pred.device)
            else:
                # 多分类
                alpha = torch.tensor([self.alpha] * pt.size(1), device=pred.device)
        elif isinstance(self.alpha, (list, tuple)) and len(self.alpha) == pt.size(1):
            alpha = torch.tensor(self.alpha, device=pred.device)
        else:
            raise ValueError("len(alpha) != num_classes")

        # 计算损失 FL = -αt * (1 - pt)^γ * log(pt)
        loss = -alpha * torch.pow(1 - pt, self.gamma) * torch.log(pt) * targets
        # 根据reduction参数返回
        if self.reduction == 'mean':
            return loss.sum() / loss.size(0)    # 按样本求平均
        else:
            return loss.sum()

class DiceLoss(nn.Module):
    """dice_loss损失"""
    def __init__(self, alpha: float|list = 1.0, smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        :param pred: 经过sigmoid/softmax的输出, 分类-(B, C) / 分割-(B, N, C)
        :param target: 未进行one-hot编码的标签, 分类-(B,) / 分割-(B, N)
        :return:
        """
        # B: batch_size, C: num_classes
        B, C = pred.size(0), pred.size(-1)
        # 模型预测概率
        probs = pred.reshape(-1, C)  # (*, C)
        # 标签one-hot编码: 多分类处理, 二分类无需编码
        if C > 1:
            tg = F.one_hot(target, num_classes=C).reshape(-1, C)
            pt = probs
        else:
            target = target.reshape(-1, 1)
            tg = torch.cat([target, 1 - target], dim=-1) # (*, 2)
            pt = torch.cat([probs, 1 - probs], dim=-1)  # (*, 2)

        # 如果alpha是标量，则扩展为与类别数量相同的向量
        if isinstance(self.alpha, float):
            if C > 1:
                alpha = torch.tensor([self.alpha] * C, device=pred.device)
            else:
                alpha = torch.tensor([self.alpha, 1 - self.alpha], device=pred.device)
        elif isinstance(self.alpha, (list, tuple)) and len(self.alpha) == C:
            alpha = torch.tensor(self.alpha, device=pred.device)
        else:
            raise ValueError("len(alpha) != num_classes")
        # 计算dice_loss
        inter_set = (pt * tg * alpha).sum()
        union_set = ((pt + tg) * alpha).sum()
        dice_loss = 1 - (2.0 * inter_set + self.smooth) / (union_set + self.smooth)
        return dice_loss

if __name__ == "__main__":
    """模块测试"""
    loss_func = LossFunc(alpha=[0.25, 0.5, 0.15, 0.1])
    B, C, N = 4, 4, 100
    for i in range(100):
        # 分类任务
        pred = torch.randint(0, 100, [B, C]) / 100
        pred_trans = torch.randint(0, 100, [B, 64, 64]) / 100
        target = torch.randint(0, C, size=(B,))
        loss = loss_func(pred, pred_trans, target)
        print(loss)
        # 分割任务
        pred = torch.randint(0, 100, [B, N, C]) / 100
        pred_trans = torch.randint(0, 100, [B, 64, 64]) / 100
        target = torch.randint(0, C, size=(B, N))
        loss = loss_func(pred, pred_trans, target)
        print(loss)
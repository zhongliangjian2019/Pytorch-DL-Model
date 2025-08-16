"""检测模型训练损失函数"""
import torch
import torch.nn.functional as F

class DetectLoss:
    """损失函数定义"""
    def __init__(self):
        pass

    def focal_loss(self, pred, target):
        """focal loss"""
        pred = pred.permute(0, 2, 3, 1)
        pos_indices = target.eq(1).float()
        neg_indices = target.lt(1).float()

        # 正样本周围赋予小权重, 聚焦中心位置
        neg_weights = torch.pow(1 - target, 4)
        # 裁剪输出, 避免梯度过大
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_indices
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_indices

        # 计算归一化损失
        num_pos = pos_indices.float().sum()
        num_neg = neg_indices.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        if num_pos == 0:
            loss = -neg_loss / num_neg
        else:
            loss = -(pos_loss / num_pos + neg_loss / num_pos)
        return loss

    def reg_l1_loss(self, heat: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """ l1 loss
        @param pred shape = Bx2xHxW
        """
        keep = (F.max_pool2d(heat, (3, 3), 1, 1) == heat).float()
        keep = keep.permute(0, 2, 3, 1).sum(dim=3, keepdim=True)
        keep = keep.repeat(1, 1, 1, 2)
        pred = pred.permute(0, 2, 3, 1)
        expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)
        expand_mask *= keep
        # 正样本趋近于真值
        pos_loss = F.l1_loss(pred * expand_mask, target * expand_mask,
                             reduction='sum') / (expand_mask.sum() / 2 + 1e-6)
        # 负样本趋近于零
        neg_loss = F.l1_loss(torch.abs(pred) * (1 - expand_mask), target * (1 - expand_mask),
                             reduction='sum') / ((1 - expand_mask).sum() / 2 + 1e-6)
        # 总损失
        loss = pos_loss + neg_loss
        return loss

if __name__ == "__main__":
    """单元测试"""
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    kernel = cv2.getGaussianKernel(5, 1.5, cv2.CV_32F)
    kernel = kernel @ kernel.T
    kernel = cv2.normalize(kernel, None, 0, 1, cv2.NORM_MINMAX)
    heat = np.zeros(shape=(128, 128), dtype=np.float32)
    radius = kernel.shape[0] // 2
    mask = np.zeros_like(heat)
    for i in range(100):
        cx = np.random.randint(kernel.shape[0] // 2, heat.shape[0] - kernel.shape[0] // 2)
        cy = np.random.randint(kernel.shape[0] // 2, heat.shape[0] - kernel.shape[0] // 2)
        heat[cy - radius: cy + radius + 1, cx - radius: cx + radius + 1] += kernel

    for i in range(100):
        cx = np.random.randint(kernel.shape[0] // 2, heat.shape[0] - kernel.shape[0] // 2)
        cy = np.random.randint(kernel.shape[0] // 2, heat.shape[0] - kernel.shape[0] // 2)
        mask[cy - radius: cy + radius + 1, cx - radius: cx + radius + 1] = 1

    heat = np.clip(heat, 0, 1)
    heat = heat[np.newaxis, np.newaxis, ...]
    heat = torch.as_tensor(heat, dtype=torch.float32)
    mask = mask[np.newaxis, ...]
    target = mask[..., np.newaxis]
    target = np.concatenate([target, target], axis=-1)
    target = torch.as_tensor(target, dtype=torch.float32)
    mask = torch.as_tensor(mask, dtype=torch.float32)
    loss = reg_l1_loss(heat, heat, target, mask)
    print(loss)



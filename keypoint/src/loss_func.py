"""模型训练损失函数"""
import torch
import torch.nn.functional as F

def focal_loss(pred, target):
    pred = pred.permute(0, 2, 3, 1)
    # -------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    # -------------------------------------------------------------------------#
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    # -------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    # -------------------------------------------------------------------------#
    neg_weights = torch.pow(1 - target, 4)

    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    # -------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    # -------------------------------------------------------------------------#
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    # -------------------------------------------------------------------------#
    #   进行损失的归一化
    # -------------------------------------------------------------------------#
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    num_neg = neg_inds.float().sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss / num_neg
    else:
        loss = -(pos_loss / num_pos + neg_loss / num_pos)

    return loss

def reg_l1_loss(pred, target, mask):
    # --------------------------------#
    #   计算l1_loss
    # --------------------------------#
    pred = pred.permute(0, 2, 3, 1)
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)
    # 正样本趋近于真值
    pos_loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum') / (mask.sum() + 1e-4)
    # 负样本趋近于零
    neg_loss = F.l1_loss(torch.abs(pred) * (1 - expand_mask), target * (1 - expand_mask), reduction='sum') / ((1 - mask).sum() + 1e-4)
    # 总损失
    loss = pos_loss + neg_loss
    return loss
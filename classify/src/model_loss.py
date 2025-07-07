"""
@brief 损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    """焦点损失"""
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, float):
            self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class PairWiseConfusionLoss(nn.Module):
    """成对混淆损失"""
    def __init__(self, weight=None):
        super(PairWiseConfusionLoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        loss = self.PairwiseConfusion(input, target)
        loss += self.EntropicConfusion(input, target, self.weight)
        return loss

    @staticmethod
    def PairwiseConfusion(features, target):
        """成对混淆损失"""
        batch_size = features.size(0)
        if float(batch_size) % 2 != 0:
            raise Exception('Incorrect batch size provided')
        batch_left = features[:int(0.5*batch_size)]
        batch_right = features[int(0.5*batch_size):]

        target_left = target[:int(0.5*batch_size)]
        target_right = target[int(0.5*batch_size):]

        target_mask_t = torch.eq(target_left, target_right)
        target_mask = 1 - target_mask_t.type(torch.float32)

        loss  = (torch.norm((batch_left - batch_right).abs(),2, 1).mul(target_mask)).sum() / batch_size

        return loss

    @staticmethod
    def EntropicConfusion(features, target, weight=None):
        """交叉熵损失"""
        if weight is None:
            weight = torch.tensor([1.0 for i in range(features.size(1))], dtype=torch.float32)
        batch_size = features.size(0)
        target = F.one_hot(torch.squeeze(target), features.size(1))
        return -(torch.mul(target, torch.log(features)) * weight).sum() * (1.0 / batch_size)

class ClassLoss(nn.Module):
    """分类模型损失函数"""
    def __init__(self, weight=None):
        super(ClassLoss, self).__init__()
        self.pc_loss = PairWiseConfusionLoss(weight)
        self.focal_loss = FocalLoss(alpha=weight)

    def forward(self, input, target):
        """
        :param input: N, C, 1, 1
        :param target: N, 1
        :return:
        """
        feature = torch.squeeze(input).softmax(dim=1)
        loss = self.pc_loss(feature, target) + self.focal_loss(input, target)
        return loss

if __name__ == "__main__":
    """模型测试"""
    input = torch.randn((2, 7, 1, 1))
    target = torch.randint(0, 7, (2, 1))
    class_loss = ClassLoss()
    class_loss(input, target)
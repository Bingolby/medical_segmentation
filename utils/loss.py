import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LossFunction(nn.Module):


    def __init__(self, weight=None):
        super(LossFunction, self).__init__()
        self.loss = CrossEntropyLoss(weight)
    def forward(self, input, target):
        return self.loss(input, target)


class CrossEntropyLoss(object):
    def __init__(self, weight=None):
        # ignored_index = 0 忽略黑色背景
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction='none').to(device)

    def __call__(self, input, target):
        n, c, h, w = input.size()
        target = target.long()
        return torch.sum(self.criterion(input, target)) / (n*h*w)


class EQLv2(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=5,  # 2 for digestpath, 5 for cervix
                 gamma=12,
                 mu=0.8,
                 alpha=4.0):
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True

        # cfg for eqlv2
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # initial variables
        self._pos_grad = None
        self._neg_grad = None
        self.pos_neg = None

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)
        print(f"build EQL v2, gamma: {gamma}, mu: {mu}, alpha: {alpha}")

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        cls_score = cls_score.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
        # print(cls_score.is_contiguous())
        # print(cls_score.shape)
        # .view(-1, self.num_classes)
        label = label.permute(0, 1, 2).view(-1).type(torch.long)
        self.n_i, self.n_c = cls_score.size()
        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target

        target = expand_label(cls_score, label)

        pos_w, neg_w = self.get_weight(cls_score)

        weight = pos_w * target + neg_w * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target,
                                                      reduction='none')
        cls_loss = torch.sum(cls_loss * weight) / self.n_i

        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())

        return self.loss_weight * cls_loss

    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)

        # do not collect grad for objectiveness branch [:-1]
        pos_grad = torch.sum(grad * target * weight, dim=0)#[:-1]
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)#[:-1]

        # dist.all_reduce(pos_grad)
        # dist.all_reduce(neg_grad)

        self._pos_grad += pos_grad
        self._neg_grad += neg_grad
        self.pos_neg = self._pos_grad / (self._neg_grad + 1e-10)

    def get_weight(self, cls_score):
        # we do not have information about pos grad and neg grad at beginning
        if self._pos_grad is None:
            self._pos_grad = cls_score.new_zeros(self.num_classes)
            self._neg_grad = cls_score.new_zeros(self.num_classes)
            neg_w = cls_score.new_ones((self.n_i, self.n_c))
            pos_w = cls_score.new_ones((self.n_i, self.n_c))
        else:
            # the negative weight for objectiveness is always 1
            # neg_w = torch.cat([self.map_func(self.pos_neg), cls_score.new_ones(1)])
            neg_w = self.map_func(self.pos_neg)
            pos_w = 1 + self.alpha * (1 - neg_w)
            neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
            pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w


if __name__ == '__main__':
    output = torch.randn(32, 2, 384, 384)
    target = torch.randint(2, (32, 384, 384))
    ce = LossFunction()
    cel = ce(output, target)
    print(cel)
    eql = EQLv2()
    eql = eql(output, target)
    print(eql)
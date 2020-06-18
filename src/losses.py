from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class OHEMLoss(nn.Module):
    def __init__(self, rate=0.8):
        super(OHEMLoss, self).__init__()
        print(f"OHEM is initialized with rate={rate}.")
        self.rate = rate

    def change_rate(self, new_rate):
        print(f"Rate Has changed! New rate is {new_rate}.")
        self.rate = new_rate

    def forward(self, cls_pred, cls_target):
        batch_size = cls_pred.size(0)
        ohem_cls_loss = F.cross_entropy(
            cls_pred, cls_target, reduction="none", ignore_index=-1
        )

        sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
        keep_num = min(sorted_ohem_loss.size()[0], int(batch_size * self.rate))
        if keep_num < sorted_ohem_loss.size()[0]:
            keep_idx_cuda = idx[:keep_num]
            ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
        cls_loss = ohem_cls_loss.sum() / keep_num
        return cls_loss


def reduced_focal_loss_with_logits(
    input: torch.Tensor,
    target: torch.Tensor,
    gamma=2.0,
    alpha: Optional[float] = None,
    reduction="mean",
    normalized=False,
    threshold: Optional[float] = 0.5,
) -> torch.Tensor:
    """Compute binary focal loss between target and output logits.

    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
    References::

        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type(input.type())

    logpt = -F.binary_cross_entropy_with_logits(input, target, reduction="none")
    pt = torch.exp(logpt)

    # compute the loss
    if threshold is None:
        focal_term = (1 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / threshold).pow(gamma)
        focal_term[pt < threshold] = 1

    loss = -focal_term * logpt

    if alpha is not None:
        loss = loss * (alpha * target + (1 - alpha) * (1 - target))

    if normalized:
        norm_factor = focal_term.sum()
        loss = loss / norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


class ReducedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, th=0.5, ignore_index=None):
        """
        Focal loss for multi-class problem.

        :param alpha:
        :param gamma:
        :param ignore_index: If not None, targets with given index are ignored
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.th = th

    def forward(self, label_input, label_target):
        num_classes = label_input.size(1)
        loss = 0

        # Filter anchors with -1 label from loss computation
        if self.ignore_index is not None:
            not_ignored = label_target != self.ignore_index

        for cls in range(num_classes):
            cls_label_target = (label_target == cls).long()
            cls_label_input = label_input[:, cls, ...]

            if self.ignore_index is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]

            loss += reduced_focal_loss_with_logits(
                cls_label_input,
                cls_label_target,
                gamma=self.gamma,
                alpha=self.alpha,
                threshold=0.5,
            )
        return loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            classes = pred.size(1)
            true_dist.fill_(self.smoothing / (classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

import torch

from torch import nn
import torch.nn.functional as F

class OHEMLoss(nn.Module):
    def __init__(self, rate=0.7):
        super(OHEMLoss, self).__init__()
        self.rate = rate

    def forward(self, cls_pred, cls_target):
        batch_size = cls_pred.size(0)
        ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)

        sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
        keep_num = min(sorted_ohem_loss.size()[0], int(batch_size * self.rate))
        if keep_num < sorted_ohem_loss.size()[0]:
            keep_idx_cuda = idx[:keep_num]
            ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
        cls_loss = ohem_cls_loss.sum() / keep_num
        return cls_loss
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm2d, Conv2d, Linear, Sequential
from torch.nn.modules.flatten import Flatten

from src.utils import Mish, bn_drop_lin


class Head(nn.Module):
    def __init__(self, nc, n, ps=0.0):
        super().__init__()
        layers = (
            [Mish(), Flatten()]
            + bn_drop_lin(nc, 512, True, ps, Mish())
            + bn_drop_lin(512, n, True, ps)
        )
        self.fc = nn.Sequential(*layers)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x):
        x = 0.5 * (F.adaptive_avg_pool2d(x, 1) + F.adaptive_max_pool2d(x, 1))
        return self.fc(x)


class AverageHead(nn.Module):
    def __init__(self, num_classes, out_features):
        super().__init__()
        #  self.pre_layers = Sequential(Conv2d(in_features, out_features, kernel_size=(1, 1), stride=(1, 1), bias=False),
        #                               BatchNorm2d(out_features, eps=1e-05, momentum=0.1,
        #                                           affine=True, track_running_stats=True))
        self.post_layers = Sequential(Flatten(), Linear(out_features, num_classes))
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x):
        #  x = self.pre_layers(x)
        x = 0.5 * (F.adaptive_avg_pool2d(x, 1) + F.adaptive_max_pool2d(x, 1))
        return self.post_layers(x)

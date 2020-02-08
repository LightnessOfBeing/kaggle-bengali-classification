import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter, Conv2d, BatchNorm2d, Sequential, Linear
from torch.nn.modules.flatten import Flatten

from src.utils import bn_drop_lin, Mish


class Head(nn.Module):
    def __init__(self, nc, n, ps=0.0):
        super().__init__()
        layers = [GeM(), Mish(), Flatten()] + \
                 bn_drop_lin(nc, 512, True, ps, Mish()) + \
                 bn_drop_lin(512, n, True, ps)
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
        return self.fc(x)


class AverageHead(nn.Module):
    def __init__(self, in_features, num_classes, out_features):
        super().__init__()
        self.pre_layers = Sequential(Conv2d(in_features, out_features, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                     BatchNorm2d(out_features, eps=1e-05, momentum=0.1,
                                                 affine=True, track_running_stats=True))
        self.post_layers = Sequential(Flatten(),
                                      Linear(out_features, num_classes))
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.pre_layers(x)
        x = 0.5 * (F.adaptive_avg_pool2d(x, 1) + F.adaptive_max_pool2d(x, 1))
        return self.post_layers(x)


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'

from typing import Optional

import torch

import torch.nn.functional as F
from cnn_finetune import make_model
from efficientnet_pytorch import EfficientNet
from torch import nn
from torch.nn.modules.flatten import Flatten

from torch.nn.parameter import Parameter

from src.utils import to_Mish, Mish


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


class MultiHeadNet(nn.Module):
    def __init__(self, encoder, pretrained, num_classes, activation):
        super().__init__()
        self.net = make_model(
            model_name=encoder,
            pretrained=pretrained,
            num_classes=1000
        )
        in_features = self.net._classifier.in_features
        if activation == "Mish":
            to_Mish(self.net)
            print("Mish activation added!")
        self.head_grapheme_root = Head(in_features, num_classes[0])
        self.head_vowel_diacritic = Head(in_features, num_classes[1])
        self.head_consonant_diacritic = Head(in_features, num_classes[2])
        print(self.net)


    def freeze(self):
        for param in self.model._features.parameters():
            param.requires_grad = False
        print("Model freezed!")

    def unfreeze(self):
        for param in self.model._features.parameters():
            param.requires_grad = True
        print("Model unfreezed!")

    def forward(self, x):
        x = self.net._features(x)
        logit_grapheme_root = self.head_grapheme_root(x)
        logit_vowel_diacritic = self.head_vowel_diacritic(x)
        logit_consonant_diacritic = self.head_consonant_diacritic(x)

        return logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic


class Efficient(nn.Module):
    def __init__(self, num_classes, encoder='efficientnet-b0', dropout=None, activation="Mish"):
        super().__init__()
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        self.net = EfficientNet.from_pretrained(encoder)
        if activation == "Mish":
            to_Mish(self.net)
            print("Mish activation added!")
        if dropout is not None:
            print("Dropout is set to 0!")
            self.net._dropout.p = 0.0
        print(self.net)

        self.head_grapheme_root = Head(n_channels_dict[encoder], num_classes[0])
        self.head_vowel_diacritic = Head(n_channels_dict[encoder], num_classes[1])
        self.head_consonant_diacritic = Head(n_channels_dict[encoder], num_classes[2])

    def freeze(self):
        for param in self.net.parameters():
            param.requires_grad = False
        print("Model freezed!")

    def unfreeze(self):
        for param in self.net.parameters():
            param.requires_grad = True
        print("Model unfreezed!")

    def forward(self, x):
        x = self.net.extract_features(x)
        logit_grapheme_root = self.head_grapheme_root(x)
        logit_vowel_diacritic = self.head_vowel_diacritic(x)
        logit_consonant_diacritic = self.head_consonant_diacritic(x)

        return logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic


def bn_drop_lin(n_in:int, n_out:int, bn:bool=True, p:float=0., actn:Optional[nn.Module]=None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers


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

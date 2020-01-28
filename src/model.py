import torch

import torch.nn.functional as F
from cnn_finetune import make_model
from efficientnet_pytorch import EfficientNet
from torch import nn

from torch.nn.parameter import Parameter


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
    def __init__(self, encoder, pretrained, num_classes, pooling):
        super().__init__()
        self.model = make_model(
            model_name=encoder,
            pretrained=pretrained,
            num_classes=1000
        )
        self.pool = None
        in_features = self.model._classifier.in_features
        self.head_grapheme_root = nn.Linear(in_features, num_classes[0])
        self.head_vowel_diacritic = nn.Linear(in_features, num_classes[1])
        self.head_consonant_diacritic = nn.Linear(in_features, num_classes[2])
        if pooling == "Gem":
            self.pool = GeM()

    def freeze(self):
        for param in self.model._features.parameters():
            param.requires_grad = False
        print("Model freezed!")

    def unfreeze(self):
        for param in self.model._features.parameters():
            param.requires_grad = True
        print("Model unfreezed!")

    def forward(self, x):
        features = self.model._features(x)
        # features = F.adaptive_avg_pool2d(features, 1)
        # DROPOUT???
        if self.pool is not None:
            features = self.pool(features)
        else:
            features = F.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)

        logit_grapheme_root = self.head_grapheme_root(features)
        logit_vowel_diacritic = self.head_vowel_diacritic(features)
        logit_consonant_diacritic = self.head_consonant_diacritic(features)

        return logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic


class Efficient(nn.Module):
    def __init__(self, num_classes, encoder='efficientnet-b0', dropout=None, pooling=None):
        super().__init__()
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        self.net = EfficientNet.from_pretrained(encoder)
        self.pool = None
        if dropout is not None:
            print("Dropout is set to 0!")
            self.net._dropout.p = 0.0

        if pooling == "Gem":
            print("GeM pooling layer is used!")
            self.pool = GeM()

        #  self.dropout_head = nn.Dropout(self.net._global_params.dropout_rate)

        self.head_grapheme_root = nn.Linear(n_channels_dict[encoder], num_classes[0])
        self.head_vowel_diacritic = nn.Linear(n_channels_dict[encoder], num_classes[1])
        self.head_consonant_diacritic = nn.Linear(n_channels_dict[encoder], num_classes[2])

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
        if self.pool is not None:
            x = self.pool(x)
        else:
            x = F.adaptive_avg_pool2d(x, 1)

        # x = self.dropout_head(x)

        x = x.view(x.size(0), -1)
        logit_grapheme_root = self.head_grapheme_root(x)
        logit_vowel_diacritic = self.head_vowel_diacritic(x)
        logit_consonant_diacritic = self.head_consonant_diacritic(x)

        return logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic

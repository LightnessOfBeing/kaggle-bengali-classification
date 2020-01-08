import torch

import torch.nn.functional as F
from cnn_finetune import make_model
from torch import nn

from torch.nn.parameter import Parameter

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class MultiHeadNet(nn.Module):
    def __init__(self, arch, pretrained, num_classes):
        super().__init__()
        #if self.source == "cnn_finetune":
        self.model = make_model(
            model_name=arch,
            pretrained=pretrained,
            num_classes=1000
        )
        in_features = self.model._classifier.in_features
        self.head_grapheme_root = nn.Linear(in_features, num_classes[0])
        self.head_vowel_diacritic = nn.Linear(in_features, num_classes[1])
        self.head_consonant_diacritic = nn.Linear(in_features, num_classes[2])
        self.gem = GeM()

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
        #features = F.adaptive_avg_pool2d(features, 1)
        features = self.gem(features)
        features = features.view(features.size(0), -1)

        logit_grapheme_root = self.head_grapheme_root(features)
        logit_vowel_diacritic = self.head_vowel_diacritic(features)
        logit_consonant_diacritic = self.head_consonant_diacritic(features)

        return logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic

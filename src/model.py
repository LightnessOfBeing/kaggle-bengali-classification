import torch.nn.functional as F
from cnn_finetune import make_model
from torch import nn

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

    def freeze(self):
        for param in self.model._features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model._features.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.model._features(x)
        features = F.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)

        logit_grapheme_root = self.head_grapheme_root(features)
        logit_vowel_diacritic = self.head_vowel_diacritic(features)
        logit_consonant_diacritic = self.head_consonant_diacritic(features)

        return logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic

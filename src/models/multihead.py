from cnn_finetune import make_model
from torch import nn

from src.models.head import Head
from src.utils import to_Mish


class MultiHeadNet(nn.Module):
    def __init__(self, encoder, pretrained, num_classes, activation):
        super().__init__()
        self.net = make_model(
            model_name=encoder, pretrained=pretrained, num_classes=1000
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

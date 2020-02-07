from efficientnet_pytorch import EfficientNet
from torch import nn

from src.models.head import Head, AverageHead
from src.utils import to_Mish
import kornia


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

        self.head_grapheme_root = AverageHead(n_channels_dict[encoder], num_classes[0])
        self.head_vowel_diacritic = AverageHead(n_channels_dict[encoder], num_classes[1])
        self.head_consonant_diacritic = AverageHead(n_channels_dict[encoder], num_classes[2])

    def freeze(self):
        for param in self.net.parameters():
            param.requires_grad = False
        print("Model freezed!")

    def unfreeze(self):
        for param in self.net.parameters():
            param.requires_grad = True
        print("Model unfreezed!")

    def custom_extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self.net._swish(self.net._bn0(self.net._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self.net._blocks):
            drop_connect_rate = self.net._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.net._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        return x

    def forward(self, x):
        x = self.custom_extract_features(x)
        logit_grapheme_root = self.head_grapheme_root(x)
        logit_vowel_diacritic = self.head_vowel_diacritic(x)
        logit_consonant_diacritic = self.head_consonant_diacritic(x)

        return logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic
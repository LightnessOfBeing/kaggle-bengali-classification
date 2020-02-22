import timm
from torch import nn

from src.models.head import AverageHead
from src.utils import to_Mish, to_ws, to_GN


class EfficientNew(nn.Module):
    def __init__(self, num_classes, encoder, activation):
        super().__init__()
        n_channels_dict = {'efficientnet_b0': (320, 1280), 'efficientnet_b1': (320, 1280), 'efficientnet_b2': (352, 1408),
                           'efficientnet_b3': 1536, 'efficientnet_b4': (448, 1792), 'efficientnet_b5': 2048,
                           'efficientnet_b6': 2304, 'efficientnet_b7': 2560}
        self.net = timm.create_model(encoder, pretrained=True)
        if activation == "Mish":
            to_Mish(self.net)
            print("Mish activation added!")
       # to_GeM(self.net)
        to_ws(self.net)
        to_GN(self.net)
        print(self.net)

        in_features = n_channels_dict[encoder][0]
        out_features = n_channels_dict[encoder][1]
        self.head_grapheme_root = AverageHead(in_features, num_classes[0], out_features)
        self.head_vowel_diacritic = AverageHead(in_features, num_classes[1], out_features)
        self.head_consonant_diacritic = AverageHead(in_features, num_classes[2], out_features)


    def forward(self, x):
        x = self.net.forward_features(x)
        logit_grapheme_root = self.head_grapheme_root(x)
        logit_vowel_diacritic = self.head_vowel_diacritic(x)
        logit_consonant_diacritic = self.head_consonant_diacritic(x)

        return logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic
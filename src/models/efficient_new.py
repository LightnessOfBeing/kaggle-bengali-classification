import timm
from torch import nn

from src.models.head import AverageHead
from src.utils import to_Mish, to_ws, to_FRN, to_GeM


class EfficientNew(nn.Module):
    def __init__(self, num_classes, encoder):
        super().__init__()
        n_channels_dict = {'efficientnet_b0': 1280, 'efficientnet_b1': 1280, 'efficientnet_b2': 1408,
                           'efficientnet_b3': 1536, 'efficientnet_b4': 1792, 'efficientnet_b5': 2048,
                           'efficientnet_b6': 2304, 'efficientnet_b7': 2560, 'seresnext50_32x4d': 2048,
                           'tf_efficientnet_b0_ns': 1280}
        self.net = timm.create_model(encoder, pretrained=True)
        to_Mish(self.net)
       # to_GeM(self.net)
       # to_ws(self.net)
       # to_FRN(self.net)
        print(self.net)

        out_features = n_channels_dict[encoder]
        self.head_grapheme_root = AverageHead(num_classes[0], out_features)
        self.head_vowel_diacritic = AverageHead(num_classes[1], out_features)
        self.head_consonant_diacritic = AverageHead(num_classes[2], out_features)


    def forward(self, x):
        x = self.net.forward_features(x)
        logit_grapheme_root = self.head_grapheme_root(x)
        logit_vowel_diacritic = self.head_vowel_diacritic(x)
        logit_consonant_diacritic = self.head_consonant_diacritic(x)

        return logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic
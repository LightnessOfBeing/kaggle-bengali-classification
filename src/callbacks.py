import numpy as np
import torch
import torch.nn.functional as F
from catalyst.dl.core import Callback, CallbackOrder, RunnerState
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt


class HMacroAveragedRecall(Callback):
    def __init__(
        self,
        input_grapheme_root_key: str = "grapheme_roots",
        input_consonant_diacritic_key: str = "consonant_diacritics",
        input_vowel_diacritic_key: str = "vowel_diacritics",

        output_grapheme_root_key: str = "logit_grapheme_root",
        output_consonant_diacritic_key: str = "logit_consonant_diacritic",
        output_vowel_diacritic_key: str = "logit_vowel_diacritic",

        prefix: str = "hmar",
    ):
        self.input_grapheme_root_key = input_grapheme_root_key
        self.input_consonant_diacritic_key = input_consonant_diacritic_key
        self.input_vowel_diacritic_key = input_vowel_diacritic_key

        self.output_grapheme_root_key = output_grapheme_root_key
        self.output_consonant_diacritic_key = output_consonant_diacritic_key
        self.output_vowel_diacritic_key = output_vowel_diacritic_key
        self.prefix = prefix

        super().__init__(CallbackOrder.Metric)

    def on_batch_end(self, state: RunnerState):
        input_grapheme_root = state.input[self.input_grapheme_root_key].detach().cpu().numpy()
        input_consonant_diacritic = state.input[self.input_consonant_diacritic_key].detach().cpu().numpy()
        input_vowel_diacritic = state.input[self.input_vowel_diacritic_key].detach().cpu().numpy()

        output_grapheme_root = state.output[self.output_grapheme_root_key]
        output_grapheme_root = F.softmax(output_grapheme_root, 1)
        _, output_grapheme_root = torch.max(output_grapheme_root, 1)
        output_grapheme_root = output_grapheme_root.detach().cpu().numpy()

        output_consonant_diacritic = state.output[self.output_consonant_diacritic_key]
        output_consonant_diacritic = F.softmax(output_consonant_diacritic, 1)
        _, output_consonant_diacritic = torch.max(output_consonant_diacritic, 1)
        output_consonant_diacritic = output_consonant_diacritic.detach().cpu().numpy()

        output_vowel_diacritic = state.output[self.output_vowel_diacritic_key]
        output_vowel_diacritic = F.softmax(output_vowel_diacritic, 1)
        _, output_vowel_diacritic = torch.max(output_vowel_diacritic, 1)
        output_vowel_diacritic = output_vowel_diacritic.detach().cpu().numpy()

        scores = []
        scores.append(recall_score(input_grapheme_root, output_grapheme_root, average='macro'))
        scores.append(recall_score(input_consonant_diacritic, output_consonant_diacritic, average='macro'))
        scores.append(recall_score(input_vowel_diacritic, output_vowel_diacritic, average='macro'))

        final_score = np.average(scores, weights=[2, 1, 1])
        state.metrics.add_batch_value(name=self.prefix, value=final_score)

class FreezeCallback(Callback):

    def __init__(self):
        super().__init__(CallbackOrder.Other)

    def on_stage_start(self, state: RunnerState):
        state.model.freeze()

class UnFreezeCallback(Callback):

    def __init__(self):
        super().__init__(CallbackOrder.Other)

    def on_stage_start(self, state: RunnerState):
        state.model.unfreeze()

class ImageViewerCallback(Callback):

    def __init__(self):
        super().__init__(CallbackOrder.Other)

    def on_stage_start(self, state: RunnerState):
        plt.imshow(state.input[0][0])
        plt.show()

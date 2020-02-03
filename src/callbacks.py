from typing import List

import catalyst
import numpy as np
import torch
import torch.nn.functional as F
from catalyst.dl import CriterionCallback, utils
from catalyst.dl.core import Callback, CallbackOrder, RunnerState
from sklearn.metrics import recall_score

from src.utils import rand_bbox


class HMacroAveragedRecall(Callback):
    def __init__(
            self,
            input_grapheme_root_key: str = "grapheme_root",
            input_consonant_diacritic_key: str = "consonant_diacritic",
            input_vowel_diacritic_key: str = "vowel_diacritic",

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
        state.metrics.add_batch_value(name="hmar_gr", value=scores[0])
        state.metrics.add_batch_value(name="hmar_cd", value=scores[1])
        state.metrics.add_batch_value(name="hmar_vd", value=scores[2])


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


class MixupCutmixCallback(CriterionCallback):
    def __init__(
            self,
            fields: List[str] = ("features",),
            alpha=1.0,
            on_train_only=True,
            weight_grapheme_root=2.0,
            weight_vowel_diacritic=1.0,
            weight_consonant_diacritic=1.0,
            **kwargs
    ):
        """
        Args:
            fields (List[str]): list of features which must be affected.
            alpha (float): beta distribution a=b parameters.
                Must be >=0. The more alpha closer to zero
                the less effect of the mixup.
            on_train_only (bool): Apply to train only.
                As the mixup use the proxy inputs, the targets are also proxy.
                We are not interested in them, are we?
                So, if on_train_only is True, use a standard output/metric
                for validation.
        """
        assert len(fields) > 0, \
            "At least one field for MixupCallback is required"
        assert alpha >= 0, "alpha must be>=0"

        super().__init__(**kwargs)

        print("Custom MixupCutmixCallback is being initialized!")
        print(f"Weights {weight_grapheme_root}, {weight_vowel_diacritic} {weight_consonant_diacritic}")

        self.on_train_only = on_train_only
        self.fields = fields
        self.alpha = alpha
        self.lam = 1
        self.index = None
        self.is_needed = True
        self.weight_grapheme_root = weight_grapheme_root
        self.weight_vowel_diacritic = weight_vowel_diacritic
        self.weight_consonant_diacritic = weight_consonant_diacritic
        self.apply_mixup = True

    def on_loader_start(self, state: RunnerState):
        self.is_needed = not self.on_train_only or \
                         state.loader_name.startswith("train")

    def do_mixup(self, state: RunnerState):

        for f in self.fields:
            state.input[f] = self.lam * state.input[f] + \
                             (1 - self.lam) * state.input[f][self.index]

    def do_cutmix(self, state: RunnerState):

        bbx1, bby1, bbx2, bby2 =\
            rand_bbox(state.input[self.fields[0]].shape, self.lam)

        for f in self.fields:
            state.input[f][:, :, bbx1:bbx2, bby1:bby2] =\
                state.input[f][self.index, :, bbx1:bbx2, bby1:bby2]

        self.lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)
                        / (state.input[self.fields[0]].shape[-1]
                           * state.input[self.fields[0]].shape[-2]))

    def on_batch_start(self, state: RunnerState):
        if not self.is_needed:
            return

        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1

        self.index = torch.randperm(state.input[self.fields[0]].shape[0])
        self.index.to(state.device)

        self.apply_mixup = (np.random.rand() < 0.5)
        if self.apply_mixup:
            self.do_mixup(state)
        else:
            self.do_cutmix(state)

    def _compute_loss(self, state: RunnerState, criterion):
        loss_arr = [0, 0, 0]
        if not self.is_needed:
            for i, (input_key, output_key) in enumerate(list(zip(self.input_key, self.output_key))):
                pred = state.output[output_key]
                y = state.input[input_key]
                loss_arr[i] = criterion(pred, y)

        else:
            for i, (input_key, output_key) in enumerate(list(zip(self.input_key, self.output_key))):
                pred = state.output[output_key]
                y_a = state.input[input_key]
                y_b = state.input[input_key][self.index]
                loss_arr[i] = self.lam * criterion(pred, y_a) + \
                              (1 - self.lam) * criterion(pred, y_b)

        loss = loss_arr[0] * self.weight_grapheme_root + \
               loss_arr[1] * self.weight_vowel_diacritic + \
               loss_arr[2] * self.weight_consonant_diacritic

        return loss


class CheckpointLoader(Callback):

    def __init__(self, checkpoint_path):
        super().__init__(CallbackOrder.Other)
        self.checkpoint_path = checkpoint_path

    def on_stage_start(self, state: RunnerState):
        print(f'Checkpoint {self.checkpoint_path} is being loaded!')
        checkpoint = utils.load_checkpoint(self.checkpoint_path)
        utils.unpack_checkpoint(checkpoint, model=state.model)


class ImageViewerCallback(Callback):

    def __init__(self):
        super().__init__(CallbackOrder.Other)

    def on_batch_start(self, state: RunnerState):
        print(type(state.input["images"].numpy()))
        img = state.input["images"].numpy()[0]
        np.savetxt("kek.txt", img[0])

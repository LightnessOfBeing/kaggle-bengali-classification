from catalyst.dl import registry
from catalyst.dl import SupervisedRunner as Runner
from pytorch_toolbelt.losses import FocalLoss, LovaszLoss

from src.dataset import BalanceSampler
from src.models.efficient_new import EfficientNew
from .models.multihead import MultiHeadNet
from .models.efficient import Efficient
from .experiment import Experiment
from .losses import *
from .callbacks import *

# Register models
registry.Model(MultiHeadNet)
registry.Model(Efficient)
registry.Model(EfficientNew)

# Register callbacks
registry.Callback(HMacroAveragedRecall)
registry.Callback(UnFreezeCallback)
registry.Callback(FreezeCallback)
registry.Callback(ImageViewerCallback)
registry.Callback(MixupCutmixCallback)
registry.Callback(CheckpointLoader)
registry.Callback(HMacroAveragedRecallSingle)
registry.Callback(MixupCutmixCallbackSingle)

# Register criterion
registry.Criterion(FocalLoss)
registry.Criterion(LovaszLoss)
registry.Criterion(OHEMLoss)
registry.Criterion(ReducedFocalLoss)

registry.Sampler(BalanceSampler)

registry.Criterion(LabelSmoothingLoss)




from catalyst.dl import SupervisedRunner as Runner
from catalyst.dl import registry
from pytorch_toolbelt.losses import FocalLoss

from src.models.efficient_new import EfficientNew

from .callbacks import *
from .experiment import Experiment
from .losses import *
from .models.efficient import Efficient
from .models.multihead import MultiHeadNet

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

# Register criterion
registry.Criterion(FocalLoss)
registry.Criterion(OHEMLoss)
registry.Criterion(ReducedFocalLoss)

registry.Criterion(LabelSmoothingLoss)

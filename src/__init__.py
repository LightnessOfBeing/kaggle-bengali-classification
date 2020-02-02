from catalyst.dl import registry
from catalyst.dl import SupervisedRunner as Runner
from pytorch_toolbelt.losses import FocalLoss, LovaszLoss

from .model import MultiHeadNet, Efficient
from .experiment import Experiment
from .losses import *
from .callbacks import *
from .optimizers import *

# Register models
registry.Model(MultiHeadNet)
registry.Model(Efficient)

# Register callbacks
registry.Callback(HMacroAveragedRecall)
registry.Callback(UnFreezeCallback)
registry.Callback(FreezeCallback)
registry.Callback(ImageViewerCallback)
registry.Callback(MixupCutmixCallback)
registry.Callback(CheckpointLoader)

# Register criterion
registry.Criterion(FocalLoss)
registry.Criterion(LovaszLoss)
registry.Criterion(OHEM_Loss)



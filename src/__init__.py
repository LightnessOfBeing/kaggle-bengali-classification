from catalyst.dl import registry
from catalyst.dl import SupervisedRunner as Runner
from pytorch_toolbelt.losses import FocalLoss, LovaszLoss

from src.dataset import BalanceSampler
from .models.multihead import MultiHeadNet
from .models.efficient import Efficient
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
registry.Callback(CutmixCallback)

# Register criterion
registry.Criterion(FocalLoss)
registry.Criterion(LovaszLoss)
registry.Criterion(OHEMLoss)
registry.Criterion(ReducedFocalLoss)

registry.Module(BalanceSampler)




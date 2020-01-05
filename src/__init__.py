from catalyst.dl import registry
from catalyst.dl import SupervisedRunner as Runner

from .model import MultiHeadNet
from .experiment import Experiment
from .losses import *
from .callbacks import *
from .optimizers import *


# Register models
registry.Model(MultiHeadNet)

# Register callbacks
registry.Callback(HMacroAveragedRecall)
registry.Callback(UnFreezeCallback)
registry.Callback(FreezeCallback)
registry.Callback(ImageViewerCallback)



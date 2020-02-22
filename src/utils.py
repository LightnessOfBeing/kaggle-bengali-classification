from typing import Optional

import torch.nn.functional as F
import torch
from efficientnet_pytorch.utils import MemoryEfficientSwish
import cv2
import numpy as np
from timm.models import SelectAdaptivePool2d
from timm.models.activations import Swish
from torch import nn
from torch.nn import AdaptiveAvgPool2d

from src.models.head import GeM


def load_image(path):
    image = cv2.imread(path, 0)
    image = np.stack((image, image, image), axis=-1)
    return image


HEIGHT = 137
WIDTH = 236
SIZE = 128
stats = (0.0692, 0.2051)


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(img0, size=SIZE, pad=16):
    # crop a box around pixels large than the threshold
    # some images contain line at the sides
    ymin, ymax, xmin, xmax = bbox(img0[5:-5, 5:-5] > 80)
    # cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax, xmin:xmax]
    # remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax - xmin, ymax - ymin
    l = max(lx, ly) + pad
    # make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l - ly) // 2,), ((l - lx) // 2,)], mode='constant')
    return cv2.resize(img, (size, size))


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def dropout_replace(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Dropout):
            setattr(model, child_name, nn.Dropout(p=0.))
        else:
            dropout_replace(child)


class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))  # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x))
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))


class Mish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)


def to_Mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU) or isinstance(child, MemoryEfficientSwish) or \
                isinstance(child, Swish):
            setattr(model, child_name, Mish())
        else:
            to_Mish(child)

def to_GeM(model):
    for child_name, child in model.named_children():
        if isinstance(child, AdaptiveAvgPool2d) or\
                isinstance(child, SelectAdaptivePool2d):
            setattr(model, child_name, GeM())
        else:
            to_GeM(child)



def bn_drop_lin(n_in: int, n_out: int, bn: bool = True, p: float = 0., actn: Optional[nn.Module] = None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers

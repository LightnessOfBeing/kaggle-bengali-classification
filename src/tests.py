import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.augmentations import shiftscalerotate_aug, valid_aug
from src.utils import HEIGHT, WIDTH, crop_resize


def load_parquet(fname):
    df = pd.read_csv(fname)
    data = 255 - df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
    image = data[1]
    image = crop_resize(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image1 = valid_aug()(image=image)["image"]
    plt.imshow(image1)
    plt.show()
    plt.imshow(image)
    plt.show()
    return image1


def load_png(fname, aug):
    print(fname)
    image = cv2.imread(fname, 0)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    print(image.shape)
    image1 = aug(image=image)["image"]
    plt.imshow(image1[:, :, 2])
    plt.show()
    return image


if __name__ == "__main__":
    for i in range(10):
        im1 = load_png(
            "../input/grapheme-imgs-128x128/Train_0_iafoss.png", shiftscalerotate_aug()
        )
        im2 = load_png(
            "../input/grapheme-imgs-128x128/Train_0.png", shiftscalerotate_aug()
        )

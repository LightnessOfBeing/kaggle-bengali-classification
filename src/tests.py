import cv2

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from albumentations import Normalize

from src.augmentations import valid_aug, simple_aug
from src.utils import HEIGHT, WIDTH, crop_resize, load_image


def visualize_image(img):
    return

def load_parquet(fname, aug):
    df = pd.read_csv(fname)
   # df.to_csv(f"{fname.split('.')[0]}.csv", index=False)
    data = 255 - df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
    image = data[1]
    image = crop_resize(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image1 = valid_aug()(image=image, mean=(0.0692, 0.0692, 0.0692), std=(0.2051, 0.2051, 0.2051))['image']
    image2 = Normalize()(image=image)['image']
    plt.imshow(image1)
    plt.show()
    plt.imshow(image)
    plt.show()
    return image1

stats = (0.0692, 0.2051)

def load_png(fname, aug):
    image = cv2.imread(fname, 0)
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    image1 = aug(image=image)['image']
    plt.imshow(image1)
    plt.show()
    return image1

if __name__ == "__main__":
    im1 = load_png("../input/grapheme-imgs-128x128/Train_0.png", simple_aug())
   # im2 = load_parquet("test1.csv", valid_aug())

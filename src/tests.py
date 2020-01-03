import cv2

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from albumentations import Normalize

from src.augmentations import train_aug
from src.utils import HEIGHT, WIDTH, crop_resize, load_image


def visualize_image(img):
    return

def load_parquet(fname):
    df = pd.read_csv(fname)
   # df.to_csv(f"{fname.split('.')[0]}.csv", index=False)
    data = 255 - df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
    img = data[1]
    img = (img * (255.0 / img.max())).astype(np.uint8)
    img = crop_resize(img)
    img = np.stack((img, img, img), axis=-1)
 #   img = train_aug()(image=img)['image'].astype(np.uint8)
    print(img.shape)
    plt.imshow(img)
    plt.show()
    return

stats = (0.0692, 0.2051)

def load_png(fname):
    img = cv2.imread(fname, 0)
    img = (img * (255.0 / img.max())).astype(np.uint8)
    img = crop_resize(img)
    img = np.stack((img, img, img), axis=-1)
   # img = train_aug()(image=img)['image'].astype(np.uint8)
    img = (img.astype(np.float32)/255.0 - stats[0]) / stats[1]
    print(img.shape)
    plt.imshow(img)
    plt.show()
    return

if __name__ == "__main__":
    load_parquet("test1.csv")
    load_png("../input/grapheme-imgs-128x128/Train_0.png")
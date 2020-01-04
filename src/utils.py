import cv2
import numpy as np

def load_image(path):
    image = cv2.imread(path, 0)
    image = np.stack((image, image, image), axis=-1)
    return image

HEIGHT = 137
WIDTH = 236
TARGET_SIZE = 256

def make_square(img, target_size=256):
    img = img[0:-1, :]
    height, width = img.shape

    x = target_size
    y = target_size

    square = np.ones((x, y), np.uint8) * 255
    square[(y - height) // 2:y - (y - height) // 2, (x - width) // 2:x - (x - width) // 2] = img

    return square


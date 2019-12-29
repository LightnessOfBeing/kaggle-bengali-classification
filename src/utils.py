import cv2
import numpy as np

def load_image(path):
    image = cv2.imread(path, 0)
    image = np.stack((image, image, image), axis=-1)
    return image
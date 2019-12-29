from albumentations import Compose, Resize, Rotate, HorizontalFlip, Normalize

def train_aug(image_size):
    return Compose([
        Resize(*image_size),
        Rotate(10),
        HorizontalFlip(),
        Normalize()
    ], p=1)


def valid_aug(image_size):
    return Compose([
        Resize(*image_size),
        Normalize()
    ], p=1)
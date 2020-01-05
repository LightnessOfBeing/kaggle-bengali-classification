from albumentations import Compose, Resize, Rotate, HorizontalFlip, Normalize

def train_aug(image_size=None):
    augs_list = [Rotate(10),
        HorizontalFlip(),
        Normalize(mean=(0.0692, 0.0692, 0.0692), std=(0.2051, 0.2051, 0.2051))]
    if image_size is not None:
        augs_list = [Resize(*image_size)] + augs_list
    return Compose(augs_list, p=1)


def valid_aug(image_size=None):
    augs_list = [Normalize(mean=(0.0692, 0.0692, 0.0692), std=(0.2051, 0.2051, 0.2051))]
    if image_size is not None:
        augs_list = [Resize(*image_size)] + augs_list
    return Compose(augs_list, p=1)
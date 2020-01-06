from albumentations import Compose, Resize, Rotate, HorizontalFlip, Normalize, VerticalFlip, ShiftScaleRotate, \
    RandomGridShuffle, Cutout, CoarseDropout


def train_aug(image_size=None):
    augs_list = [HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(scale_limit=0.2, rotate_limit=45, shift_limit=0.15, p=0.7, border_mode=0),
        RandomGridShuffle(grid=(2, 2), p=0.5),
        CoarseDropout(max_holes=10, max_height=5, max_width=5, fill_value=255, p=0.5),
        Normalize(mean=(0.0692, 0.0692, 0.0692), std=(0.2051, 0.2051, 0.2051))]
    if image_size is not None:
        augs_list = [Resize(*image_size)] + augs_list
    return Compose(augs_list, p=1)


def valid_aug(image_size=None):
    augs_list = [Normalize(mean=(0.0692, 0.0692, 0.0692), std=(0.2051, 0.2051, 0.2051))]
    if image_size is not None:
        augs_list = [Resize(*image_size)] + augs_list
    return Compose(augs_list, p=1)
from albumentations import Compose, Resize, Rotate, HorizontalFlip, Normalize, VerticalFlip, ShiftScaleRotate, \
    RandomGridShuffle, Cutout, CoarseDropout


def simple_aug(image_size=None):
    augs_list = [
        Rotate(20),
        CoarseDropout(max_holes=20, max_height=5, max_width=5, fill_value=255, p=0.5),
        Normalize(mean=(0.0692, 0.0692, 0.0692), std=(0.2051, 0.2051, 0.2051))]
    if image_size is not None:
        augs_list = [Resize(*image_size)] + augs_list
    return Compose(augs_list, p=1)


def mixup_aug(image_size=None):
    augs_list = [HorizontalFlip(p=0.3), Normalize(mean=(0.0692, 0.0692, 0.0692), std=(0.2051, 0.2051, 0.2051))]
    if image_size is not None:
        augs_list = [Resize(*image_size)] + augs_list
    return Compose(augs_list, p=1)


def valid_aug(image_size=None):
    augs_list = [Normalize(mean=(0.0692, 0.0692, 0.0692), std=(0.2051, 0.2051, 0.2051))]
    if image_size is not None:
        augs_list = [Resize(*image_size)] + augs_list
    return Compose(augs_list, p=1)


def get_augmentation(aug_name, image_size=None):
    aug_dict = {
        'valid_aug': valid_aug,
        'mixup_aug': mixup_aug,
        'simple_aug': simple_aug
    }
    return aug_dict[aug_name](image_size)

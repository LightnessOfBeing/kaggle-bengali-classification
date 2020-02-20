from albumentations import Compose, Resize, Rotate, HorizontalFlip, Normalize, VerticalFlip, ShiftScaleRotate, \
    RandomGridShuffle, Cutout, CoarseDropout
from albumentations.pytorch import ToTensorV2


def simple_aug():
    augs_list = [
        CoarseDropout(min_holes=2, max_holes=10, max_height=10, max_width=10, fill_value=0, p=0.5),
        Normalize(mean=(0.0692, 0.0692, 0.0692), std=(0.2051, 0.2051, 0.2051)),
        ToTensorV2()]
    return Compose(augs_list, p=1)


def mixup_aug():
    augs_list = [Normalize(mean=(0.0692, 0.0692, 0.0692), std=(0.2051, 0.2051, 0.2051)),
                 ToTensorV2()]
    return Compose(augs_list, p=1)


def valid_aug():
    augs_list = [Normalize(mean=(0.0692, 0.0692, 0.0692), std=(0.2051, 0.2051, 0.2051)),
                 ToTensorV2()]
    return Compose(augs_list, p=1)


def get_augmentation(aug_name):
    aug_dict = {
        'valid_aug': valid_aug,
        'mixup_aug': mixup_aug,
        'simple_aug': simple_aug
    }
    return aug_dict[aug_name]()

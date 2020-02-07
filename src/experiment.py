import os
from collections import OrderedDict

from catalyst.dl import ConfigExperiment
from sklearn.model_selection import train_test_split

from src.augmentations import get_augmentation
from src.dataset import BengaliDataset
import pandas as pd


class Experiment(ConfigExperiment):

    def get_datasets(self, stage: str, **kwargs):
        train_csv_name = kwargs.get('train_csv_name', "train.csv")
        train_csv_path = kwargs.get('train_csv_path', "../input/bengaliai-cv19/")
        df = pd.read_csv(os.path.join(train_csv_path, train_csv_name))
        data_folder = kwargs.get('data_folder', None)
        test_run = kwargs.get('test_run', None)
        train_aug_name = kwargs.get('train_aug_name', None)
        processed = kwargs.get('processed', None)
        valid_fold = kwargs.get('fold', None)
        if test_run:
            df = df[:2048]
        datasets = OrderedDict()
        if train_csv_name:
            train_transform = get_augmentation(train_aug_name)
            valid_transform = get_augmentation('valid_aug')
            if valid_fold is None:
                train_df, valid_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=65)
            else:
                print(f"Using fold {valid_fold} for validation!")
                train_df = df[df['fold'] != valid_fold]
                valid_df = df[df['fold'] == valid_fold]
            if processed:
                print("Processed images are used!")
            train_set = BengaliDataset(
                df=train_df,
                transform=train_transform,
                data_folder=data_folder,
                processed=processed
            )
            valid_set = BengaliDataset(
                df=valid_df,
                transform=valid_transform,
                data_folder=data_folder,
                processed=processed
            )
            datasets["train"] = train_set
            datasets["valid"] = valid_set
        return datasets

    @staticmethod
    def get_transforms(stage: str = None, mode: str = None, **kwargs):
        train_aug_name = kwargs.get('train_aug_name', None)
        if mode == "train":
            return get_augmentation(train_aug_name)
        return get_augmentation('valid')
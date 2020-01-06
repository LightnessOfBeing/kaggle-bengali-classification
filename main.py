#!/usr/bin/env python3
import cv2
import os

import click
import numpy as np
import pandas as pd
import torch
from cnn_finetune import make_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.augmentations import valid_aug
from src.dataset import BengaliDataset
from src.utils import HEIGHT, WIDTH, crop_resize, stats, load_image
from src.model import MultiHeadNet

class GraphemeDatasetTest(Dataset):
    def __init__(self, fname, transform):
        self.transform = transform
        self.df = pd.read_parquet(fname)
        self.data = self.df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.df.iloc[idx, 0]
        image = self.data[idx]
        image = (image * (255.0 / image.max())).astype(np.uint8)
        image = crop_resize(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if self.transform:
            image = self.transform(image=image)['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return image, name


TEST = ['test_image_data_0.parquet',
        'test_image_data_1.parquet',
        'test_image_data_2.parquet',
        'test_image_data_3.parquet']

log_dir = "logs/bengali_logs"

device = torch.device('cuda')

@click.command()
@click.option("--data_folder", type=str, default="../input/bengaliai-cv19/")
@click.option("--weights_path", type=str, default="{log_dir}/checkpoints/best.pth")
@click.option("--arch", type=str, default="resnet18")
@click.option("--sub_name", type=str, default="submission.csv")
@click.option("--bs", type=int, default=128)
@click.option("--num_workers", type=int, default=2)
def predict(data_folder, weights_path, arch, sub_name, bs, num_workers):
    row_id, target, actual = [], [], []
    model = MultiHeadNet(arch, True, [168, 11, 7])
    checkpoint = torch.load(weights_path)
    #checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    df = pd.read_csv(data_folder + "train.csv")[:20]
    ds = BengaliDataset(df, data_folder, valid_aug())
    dl = DataLoader(ds, batch_size=bs, num_workers=num_workers, shuffle=False)
    with torch.no_grad():
        for x, y in tqdm(dl):
            p1, p2, p3 = model(x.cuda())
            # p1, p2, p3 = model(x)
            p1 = p1.argmax(-1).view(-1).cpu()
            p2 = p2.argmax(-1).view(-1).cpu()
            p3 = p3.argmax(-1).view(-1).cpu()
            for idx, name in enumerate(y):
                row_id += [f'{name}_grapheme_root', f'{name}_vowel_diacritic',
                           f'{name}_consonant_diacritic']
                target += [p1[idx].item(), p2[idx].item(), p3[idx].item()]
        sub_df = pd.DataFrame({'row_id': row_id, 'target': target})
        sub_df.to_csv("submission_png.csv", index=False)
        sub_df.head()
    print("png done")

    for fname in ['train_image_data_0.parquet']:
        ds = GraphemeDatasetTest(data_folder + fname, valid_aug())[:20]
        dl = DataLoader(ds, batch_size=bs, num_workers=num_workers, shuffle=False)
        with torch.no_grad():
            for x, y in tqdm(dl):
                p1, p2, p3 = model(x.cuda())
               # p1, p2, p3 = model(x)
                p1 = p1.argmax(-1).view(-1).cpu()
                p2 = p2.argmax(-1).view(-1).cpu()
                p3 = p3.argmax(-1).view(-1).cpu()
                for idx, name in enumerate(y):
                    row_id += [f'{name}_grapheme_root', f'{name}_vowel_diacritic',
                               f'{name}_consonant_diacritic']
                    target += [p1[idx].item(), p2[idx].item(), p3[idx].item()]

    sub_df = pd.DataFrame({'row_id': row_id, 'target': target})
    sub_df.to_csv("submission_parquet.csv", index=False)
    sub_df.head()


    return

if __name__ == "__main__":
    '''
    image_id = os.path.join("../input/grapheme-imgs-128x128", "Train_0.png")
    model = make_model(
        model_name='resnet18',
        pretrained=False,
        num_classes=1000
    )
    print(dir(model._classifier.modules))
    img = load_image(imge_id)
    image = valid_aug()(image=image)
    plt.imshow(image)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    '''
    predict()
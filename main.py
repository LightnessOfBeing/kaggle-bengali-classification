#!/usr/bin/env python3
import click
import numpy as np
import pandas as pd
import torch
from scipy.stats import stats
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils import HEIGHT, WIDTH, crop_resize, stats
from src.model import MultiHeadNet

class GraphemeDatasetTest(Dataset):
    def __init__(self, fname):
        self.df = pd.read_parquet(fname)
        self.data = 255 - self.df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.df.iloc[idx,0]
        #normalize each image by its max val
        img = (self.data[idx]*(255.0/self.data[idx].max())).astype(np.uint8)
        img = crop_resize(img)
        img = (img.astype(np.float32)/255.0 - stats[0])/stats[1]
        return img, name


TEST = ['test_image_data_0.parquet',
        'test_image_data_1.parquet',
        'test_image_data_2.parquet',
        'test_image_data_3.parquet']

log_dir = "./logs/bengali_logs"

device = torch.device('cuda')

@click.command()
@click.option("--data_folder", type=str, default="../input/bengaliai-cv19/")
@click.option("--weights_name", type=str, default="best.pth")
@click.option("--arch", type=str, default="resnet34")
@click.option("--sub_name", type=str, default="submission.csv")
@click.option("--bs", type=int, default=128)
@click.option("--num_workers", type=int, default=2)
def predict(data_folder, weights_name, arch, sub_name, bs, num_workers):
    row_id, target = [], []
    model = MultiHeadNet(arch, True, [168, 11, 7])
    checkpoint = f"{log_dir}/checkpoints/{weights_name}"
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    for fname in TEST:
        ds = GraphemeDatasetTest(data_folder + fname)
        dl = DataLoader(ds, batch_size=bs, num_workers=num_workers, shuffle=False)
        with torch.no_grad():
            for x, y in tqdm(dl):
                x = x.unsqueeze(1).cuda()
                p1, p2, p3 = model(x)
                p1 = p1.argmax(-1).view(-1).cpu()
                p2 = p2.argmax(-1).view(-1).cpu()
                p3 = p3.argmax(-1).view(-1).cpu()
                for idx, name in enumerate(y):
                    row_id += [f'{name}_grapheme_root', f'{name}_vowel_diacritic',
                               f'{name}_consonant_diacritic']
                    target += [p1[idx].item(), p2[idx].item(), p3[idx].item()]

    sub_df = pd.DataFrame({'row_id': row_id, 'target': target})
    sub_df.to_csv(sub_name, index=False)
    sub_df.head()
    return

if __name__ == "__main__":
    predict()
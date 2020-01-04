#!/usr/bin/env python3

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.augmentations import valid_aug
from src.model import MultiHeadNet
from src.utils import HEIGHT, WIDTH, make_square, TARGET_SIZE


class BengaliParquetDataset(Dataset):

    def __init__(self, parquet_file, transform=None):
        self.data = pd.read_parquet(parquet_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tmp = self.data.iloc[idx, 1:].values.reshape(HEIGHT, WIDTH)
        img = np.zeros((TARGET_SIZE, TARGET_SIZE, 3))
        img[..., 0] = make_square(tmp, target_size=TARGET_SIZE)
        img[..., 1] = img[..., 0]
        img[..., 2] = img[..., 0]

        image_id = self.data.iloc[idx, 0]

        if self.transform:
            img = self.transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)

        return img, image_id

TEST = ['test_image_data_0.parquet',
        'test_image_data_1.parquet',
        'test_image_data_2.parquet',
        'test_image_data_3.parquet']

log_dir = "logs/bengali_logs"

device = torch.device('cuda')

@click.command()
@click.option("--data_folder", type=str, default="../input/bengaliai-cv19/")
@click.option("--weights_name", type=str, default="best.pth")
@click.option("--arch", type=str, default="resnet18")
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
        ds = BengaliParquetDataset(data_folder + fname, valid_aug())
        ds.__getitem__(0)
        dl = DataLoader(ds, batch_size=bs, num_workers=num_workers, shuffle=False)
        with torch.no_grad():
            for x, y in tqdm(dl):
                p1, p2, p3 = model(x.cuda())
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
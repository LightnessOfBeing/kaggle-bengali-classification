import os

import cv2
import numpy as np
from torch.utils.data import Dataset


class BengaliDataset(Dataset):

    def __init__(self, df, data_folder, transform, processed=False):
        self.image_ids = df['image_id'].values
        self.grapheme_roots = df['grapheme_root'].values
        self.vowel_diacritics = df['vowel_diacritic'].values
        self.consonant_diacritics = df['consonant_diacritic'].values
        self.data_folder = data_folder
        self.transform = transform
        self.processed = processed
        print(f"Processed {self.processed}")
        if self.processed:
            self.array = np.zeros((len(self.image_ids), 128, 128), dtype=np.uint8)
            self._populate_array()

    def _populate_array(self):
        print(len(self.image_ids), self.array.shape[0])
        assert len(self.image_ids) == self.array.shape[0]
        print("Array population started!")
        for index, image_id in enumerate(self.image_ids):
            image_path = os.path.join(self.data_folder, image_id + '.png')
            image = cv2.imread(image_path, 0).astype(np.uint8)
            self.array[index, ...] = image
        print("Array population finished!")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        grapheme_root = self.grapheme_roots[idx]
        vowel_diacritic = self.vowel_diacritics[idx]
        consonant_diacritic = self.consonant_diacritics[idx]

        if not self.processed:
            image_path = os.path.join(self.data_folder, image_id + '.png')
            image = cv2.imread(image_path, 0)
        else:
            image = self.array[idx, ...]
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        #image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            'image': image,
            'name': image_id,
            'grapheme_root': grapheme_root,
            'vowel_diacritic': vowel_diacritic,
            'consonant_diacritic': consonant_diacritic
        }

'''
TASK = {
    'grapheme_root': {'num_class':168},
    'vowel_diacritic': {'num_class':11},
    'consonant_diacritic': {'num_class':168},
    'grapheme': {
        'num_class':1295,
        'class_map': dict(pd.read_csv("../input/bengaliutils2/grapheme_1295.csv")[['grapheme', 'label']].values),
    }
}

NUM_TASK = len(TASK)
NUM_CLASS = [TASK[k]['num_class'] for k in ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'grapheme']]

class BalanceSampler(Sampler):
    def __init__(self, df_path="../input/bengaliutils2/train_with_fold.csv"):
        print("Balance sampler is inited!")
        df = pd.read_csv(df_path)
        df = df[df['fold'] != 0]
        self.length = len(df)
        df = df.reset_index()

        group = []
        grapheme_gb = df.groupby(['grapheme'])
        for k, i in TASK['grapheme']['class_map'].items():
            g = grapheme_gb.get_group(k).index
            group.append(list(g))
            assert (len(g) > 0)
        self.group = group
        print("Balance sampler is inited!")

    def __iter__(self):
        # l = iter(range(self.num_samples))
        # return l

        # for i in range(self.num_sample):
        #     yield i

        index = []
        n = 0

        is_loop = True
        while is_loop:
            num_class = TASK['grapheme']['num_class']  # 1295
            c = np.arange(num_class)
            np.random.shuffle(c)
            for t in c:
                i = np.random.choice(self.group[t])
                index.append(i)
                n += 1
                if n == self.length:
                    is_loop = False
                    break
        return iter(index)

    def __len__(self):
        return self.length
        '''
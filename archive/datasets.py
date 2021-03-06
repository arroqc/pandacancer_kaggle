import torch
import torch.utils.data as tdata
import torchvision.transforms as transforms
import PIL.Image as Image
from pathlib import Path


class TileDataset(tdata.Dataset):

    def __init__(self, img_path, dataframe, num_tiles, suffix, transform=None, tiles_transform=None, one_hot=False):

        self.suffix = suffix
        self.img_path = Path(img_path)
        self.one_hot = one_hot
        self.df = dataframe
        self.num_tiles = num_tiles
        self.img_list = self.df['image_id'].values
        self.transform = transform
        self.tiles_transform = tiles_transform

    def __getitem__(self, idx):
        img_id = self.img_list[idx]

        tiles = [str(self.img_path/(img_id + '_' + str(i) + self.suffix + '.png')) for i in range(0, self.num_tiles)]
        # vals = self.value_df[self.value_df['image_id'] == img_id]
        # tile_sample = vals.sample(self.num_tiles, weights=vals['value'] + 1e-4)['filename']
        # tiles = [str(self.img_path / tile_fn) for tile_fn in tile_sample]
        metadata = self.df.iloc[idx]
        image_tiles = []

        for tile in tiles:
            image = Image.open(tile)

            if self.transform is not None:
                image = self.transform(image)

            image = 1 - image
            # image = transforms.Normalize([1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304],
            #                              [0.36357649, 0.49984502, 0.40477625])(image)
            image_tiles.append(image)

        if torch.is_tensor(image_tiles[0]):
            image_tiles = torch.stack(image_tiles, dim=0)

        if self.tiles_transform is not None:
            image_tiles = self.tiles_transform(image_tiles)

        isup = metadata['isup_grade']
        if self.one_hot:
            if isup == 0:
                isup = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
            elif isup == 1:
                isup = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)
            elif isup == 2:
                isup = torch.tensor([1, 1, 0, 0, 0], dtype=torch.float32)
            elif isup == 3:
                isup = torch.tensor([1, 1, 1, 0, 0], dtype=torch.float32)
            elif isup == 4:
                isup = torch.tensor([1, 1, 1, 1, 0], dtype=torch.float32)
            elif isup == 5:
                isup = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32)

        return {'image': image_tiles, 'provider': metadata['data_provider'],
                'isup': isup, 'gleason': metadata['gleason_score']}

    def __len__(self):
        return len(self.img_list)


import numpy as np
class SquareDataset(tdata.Dataset):

    def __init__(self, img_path, dataframe, num_tiles, suffix, transform=None, tiles_transform=None, one_hot=False):

        self.suffix = suffix
        self.img_path = Path(img_path)
        self.one_hot = one_hot
        self.df = dataframe
        self.num_tiles = 36
        self.img_list = self.df['image_id'].values
        self.transform = transform

    def __getitem__(self, idx):
        img_id = self.img_list[idx]

        tiles = [str(self.img_path/(img_id + '_' + str(i) + self.suffix + '.png')) for i in range(0, self.num_tiles)]
        tiles = [np.array(Image.open(tile)) for tile in tiles]
        # vals = self.value_df[self.value_df['image_id'] == img_id]
        # tile_sample = vals.sample(self.num_tiles, weights=vals['value'] + 1e-4)['filename']
        # tiles = [str(self.img_path / tile_fn) for tile_fn in tile_sample]
        metadata = self.df.iloc[idx]

        idxes = list(range(36))
        image_size = 224
        images = np.zeros((image_size * 6, image_size * 6, 3))
        for h in range(6):
            for w in range(6):
                i = h * 6 + w

                this_img = tiles[idxes[i]]

                this_img = 255 - this_img
                if self.transform is not None:
                    this_img = self.transform(image=this_img)['image']
                h1 = h * image_size
                w1 = w * image_size
                images[h1:h1 + image_size, w1:w1 + image_size] = this_img

        if self.transform is not None:
            images = self.transform(image=images)['image']

        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)

        isup = metadata['isup_grade']
        if self.one_hot:
            if isup == 0:
                isup = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
            elif isup == 1:
                isup = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)
            elif isup == 2:
                isup = torch.tensor([1, 1, 0, 0, 0], dtype=torch.float32)
            elif isup == 3:
                isup = torch.tensor([1, 1, 1, 0, 0], dtype=torch.float32)
            elif isup == 4:
                isup = torch.tensor([1, 1, 1, 1, 0], dtype=torch.float32)
            elif isup == 5:
                isup = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32)

        return {'image': torch.tensor(images), 'provider': metadata['data_provider'],
                'isup': isup, 'gleason': metadata['gleason_score']}

    def __len__(self):
        return len(self.img_list)


# import pandas as pd
# values = pd.read_csv("D:\Datasets\panda/files_256_2.csv")
# img_id = '000920ad0b612851f8e01bcc880d9b3d'
# vals = values[values['image_id'] == img_id]
# print(vals.sample(16, weights=vals['value'])['filename'])


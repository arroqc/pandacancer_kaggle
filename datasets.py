import torch
import torch.utils.data as tdata
import torchvision.transforms as transforms
import PIL.Image as Image
from pathlib import Path


class TileDataset(tdata.Dataset):

    def __init__(self, img_path, dataframe, num_tiles, suffix, transform=None, tiles_transform=None):

        self.suffix = suffix
        self.img_path = Path(img_path)
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
            image = transforms.Normalize([1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304],
                                         [0.36357649, 0.49984502, 0.40477625])(image)
            image_tiles.append(image)

        if torch.is_tensor(image_tiles[0]):
            image_tiles = torch.stack(image_tiles, dim=0)

        if self.tiles_transform is not None:
            image_tiles = self.tiles_transform(image_tiles)

        return {'image': image_tiles, 'provider': metadata['data_provider'],
                'isup': metadata['isup_grade'], 'gleason': metadata['gleason_score']}

    def __len__(self):
        return len(self.img_list)

# import pandas as pd
# values = pd.read_csv("D:\Datasets\panda/files_256_2.csv")
# img_id = '000920ad0b612851f8e01bcc880d9b3d'
# vals = values[values['image_id'] == img_id]
# print(vals.sample(16, weights=vals['value'])['filename'])


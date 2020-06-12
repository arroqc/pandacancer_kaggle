import numpy as np
import torch.utils.data as tdata
from pathlib import Path
from PIL import Image
import torch


class TileDataset(tdata.Dataset):
    def __init__(self, img_path, dataframe, num_tiles, suffix,
                 transform=None,
                 one_hot=False,
                 return_stitched=True):

        self.suffix = suffix
        self.img_path = Path(img_path)
        self.one_hot = one_hot
        self.df = dataframe
        self.num_tiles = num_tiles
        self.img_list = self.df['image_id'].values
        self.transform = transform
        self.return_stitched = return_stitched

    def __getitem__(self, idx):
        img_id = self.img_list[idx]

        tiles_paths = [str(self.img_path/(img_id + '_' + str(i) + self.suffix + '.png'))
                       for i in range(0, self.num_tiles)]
        tiles = [np.array(Image.open(tile)) for tile in tiles_paths]
        metadata = self.df.iloc[idx]

        if self.return_stitched:
            images = self.make_square(tiles)
        else:
            images = self.make_bag(tiles)

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
        else:
            isup = float(isup)

        return {'image': images, 'provider': metadata['data_provider'],
                'isup': isup, 'gleason': metadata['gleason_score']}

    def __len__(self):
        return len(self.img_list)

    def make_square(self, tiles):
        idxes = list(range(self.num_tiles))
        image_size = tiles[0].shape[0]

        n_rows = int(self.num_tiles ** 0.5)
        images = np.zeros((image_size * n_rows, image_size * n_rows, 3))
        for h in range(n_rows):
            for w in range(n_rows):
                i = h * n_rows + w

                this_img = tiles[idxes[i]]

                this_img = 255 - this_img
                if self.transform is not None:
                    this_img = self.transform(image=this_img)['image']  # albumentations

                h1 = h * image_size
                w1 = w * image_size
                images[h1:h1 + image_size, w1:w1 + image_size] = this_img

        if self.transform is not None:
            images = self.transform(image=images)['image']  # albumentations

        images = images.astype(np.float32)
        images = images.transpose(2, 0, 1)
        images /= 255
        images = torch.tensor(images)
        return images

    def make_bag(self, tiles):
        images = []
        for tile in tiles:
            tile = 255 - tile
            if self.transform is not None:
                tile = self.transform(image=tile)['image']
            tile = tile.astype(np.float32)
            images.append(tile)
        images = np.stack(images, axis=0)
        images = images.astype(np.float32)
        images = images.transpose(0, 3, 1, 2)
        images /= 255
        images = torch.tensor(images)
        return images

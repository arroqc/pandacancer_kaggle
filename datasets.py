import torch
import torch.utils.data as tdata
import torchvision.transforms as transforms
import PIL.Image as Image
from pathlib import Path


class TileDataset(tdata.Dataset):

    def __init__(self, img_path, dataframe, num_tiles, transform=None, normalize_stats=None):

        self.img_path = Path(img_path)
        self.df = dataframe
        self.num_tiles = num_tiles
        self.img_list = self.df['image_id'].values
        self.transform = transform
        if normalize_stats is not None:
            self.normalize_stats = {}
            for k, v in normalize_stats.items():
                self.normalize_stats[k] = transforms.Normalize(v[0], v[1])
        else:
            self.normalize_stats = None

    def __getitem__(self, idx):
        img_id = self.img_list[idx]

        tiles = [str(self.img_path/(img_id + '_' + str(i) + '.png')) for i in range(0, self.num_tiles)]
        metadata = self.df.iloc[idx]
        image_tiles = []

        for tile in tiles:
            image = Image.open(tile)

            if self.transform is not None:
                image = self.transform(image)

            if self.normalize_stats is not None:
                image = 1 - image
                image = transforms.Normalize([1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304],
                                             [0.36357649, 0.49984502, 0.40477625])(image)
                # provider = metadata['data_provider']
                # image = self.normalize_stats[provider](image)
            image_tiles.append(image)

        image_tiles = torch.stack(image_tiles, dim=0)

        return {'image': image_tiles, 'provider': metadata['data_provider'],
                'isup': metadata['isup_grade'], 'gleason': metadata['gleason_score']}

    def __len__(self):
        return len(self.img_list)

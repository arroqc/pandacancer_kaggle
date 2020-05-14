import skimage.io
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", default='G:/Datasets/panda', required=False)
parser.add_argument("--out_dir", default='D:/Datasets/panda', required=False)
args = parser.parse_args()

BASE_PATH = Path(args.base_dir)
OUTPUT_BASE = Path(args.out_dir)


class TileMaker:

    def __init__(self, size, number):
        self.size = size
        self.number = number

    def make(self, image, mask):
        h, w, c = image.shape
        horizontal_pad = 0 if (w % self.size) == 0 else self.size - (w % self.size)
        vertical_pad = 0 if (h % self.size) == 0 else self.size - (h % self.size)

        image = np.pad(image, pad_width=((vertical_pad // 2, vertical_pad - vertical_pad // 2),
                                         (horizontal_pad // 2, horizontal_pad - horizontal_pad // 2),
                                         (0, 0)),
                       mode='constant', constant_values=255)  # Empty is white in this data

        mask = np.pad(mask, pad_width=((vertical_pad // 2, vertical_pad - vertical_pad // 2),
                                       (horizontal_pad // 2, horizontal_pad - horizontal_pad // 2),
                                       (0, 0)),
                      mode='constant', constant_values=0)  # Empty is black in this data

        h, w, c = image.shape
        image = image.reshape(h // self.size, self.size, w // self.size, self.size, c)
        image = image.swapaxes(1, 2).reshape(-1, self.size, self.size, c)
        mask = mask.reshape(h // self.size, self.size, w // self.size, self.size, c)
        mask = mask.swapaxes(1, 2).reshape(-1, self.size, self.size, c)

        if image.shape[0] < self.number:
            image = np.pad(image, pad_width=((0, self.number - image.shape[0]), (0, 0), (0, 0), (0, 0)),
                           mode='constant', constant_values=255)
            mask = np.pad(mask, pad_width=((0, self.number - mask.shape[0]), (0, 0), (0, 0), (0, 0)),
                          mode='constant', constant_values=0)

        # Find the images with the most stuff (the most red):
        sorted_tiles = np.argsort(np.sum(image[:, :, :, 0:1], axis=(1, 2, 3)))
        sorted_tiles = sorted_tiles[:self.number]

        return image[sorted_tiles], mask[sorted_tiles]


TRAIN_PATH = BASE_PATH/'train_images/'
MASKS_TRAIN_PATH = BASE_PATH/'train_label_masks/'
OUTPUT_IMG_PATH = OUTPUT_BASE/'train_tiles_256_2/imgs/'
OUTPUT_MASK_PATH = OUTPUT_BASE/'train_tiles_256_2/masks/'
CSV_PATH = BASE_PATH/'train.csv'
LEVEL = -2

OUTPUT_IMG_PATH.mkdir(exist_ok=True, parents=True)
OUTPUT_MASK_PATH.mkdir(exist_ok=True, parents=True)

tile_maker = TileMaker(256, 12)

img_list = list(TRAIN_PATH.glob('**/*.tiff'))
# img_list.pop(5765)
bad_images = []
bad_masks = []
image_stats = []
for i, img_fn in enumerate(img_list):

    img_id = img_fn.stem
    mask_fn = MASKS_TRAIN_PATH / (img_id + '_mask.tiff')

    try:
        col = skimage.io.MultiImage(str(img_fn))
        image = col[LEVEL]
    except:
        bad_images.append(img_id)
        continue

    if mask_fn.exists():

        try:
            mask = skimage.io.MultiImage(str(mask_fn))[LEVEL]
        except:
            bad_masks.append(img_id)
            mask = np.zeros_like(image)

    else:
        mask = np.zeros_like(image)

    image, mask = tile_maker.make(image, mask)
    sys.stdout.write(f'\r{i + 1}/{len(img_list)}')

    image_stats.append({'image_id': img_id, 'mean': image.mean(axis=(0, 1, 2)) / 255,
                        'mean_square': ((image / 255) ** 2).mean(axis=(0, 1, 2))})

    for i, (tile_image, tile_mask) in enumerate(zip(image, mask)):
        skimage.io.imsave(OUTPUT_IMG_PATH / (img_id + '_' + str(i) + '.png'), tile_image, check_contrast=False)
        skimage.io.imsave(OUTPUT_MASK_PATH / (img_id + '_' + str(i) + '.png'), tile_mask, check_contrast=False)

image_stats = pd.DataFrame(image_stats)
df = pd.read_csv(CSV_PATH)
df = pd.merge(df, image_stats, on='image_id', how='left')
provider_stats = {}
for provider in df['data_provider'].unique():
    mean = (df[df['data_provider'] == provider]['mean']).mean()
    std = np.sqrt((df[df['data_provider'] == provider]['mean_square']).mean() - mean ** 2)
    provider_stats[provider] = (mean, std)
provider_stats['all'] = (mean, std)
with open(OUTPUT_BASE/'stats.pkl', 'wb') as file:
    pickle.dump(provider_stats, file)

print(bad_images)
print(bad_masks)
print(provider_stats)

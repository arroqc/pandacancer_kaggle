import skimage.io
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import pickle
import argparse
import cv2


parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", default='G:/Datasets/panda', required=False)
parser.add_argument("--out_dir", default='D:/Datasets/panda', required=False)
args = parser.parse_args()

BASE_PATH = Path(args.base_dir)
OUTPUT_BASE = Path(args.out_dir)

SIZE = 128
NUM = 16
LEVEL = 1
STRIDE = False
TRAIN_PATH = BASE_PATH/'train_images/'
MASKS_TRAIN_PATH = BASE_PATH/'train_label_masks/'
OUTPUT_IMG_PATH = OUTPUT_BASE/f'train_tiles_{SIZE}_{LEVEL}/imgs/'
OUTPUT_MASK_PATH = OUTPUT_BASE/f'train_tiles_{SIZE}_{LEVEL}/masks/'
PICKLE_NAME = OUTPUT_BASE/f'stats_{SIZE}_{LEVEL}.pkl'
CSV_PATH = BASE_PATH/'train.csv'


pen_marked_images = [
    'fd6fe1a3985b17d067f2cb4d5bc1e6e1',
    'ebb6a080d72e09f6481721ef9f88c472',
    'ebb6d5ca45942536f78beb451ee43cc4',
    'ea9d52d65500acc9b9d89eb6b82cdcdf',
    'e726a8eac36c3d91c3c4f9edba8ba713',
    'e90abe191f61b6fed6d6781c8305fe4b',
    'fd0bb45eba479a7f7d953f41d574bf9f',
    'ff10f937c3d52eff6ad4dd733f2bc3ac',
    'feee2e895355a921f2b75b54debad328',
    'feac91652a1c5accff08217d19116f1c',
    'fb01a0a69517bb47d7f4699b6217f69d',
    'f00ec753b5618cfb30519db0947fe724',
    'e9a4f528b33479412ee019e155e1a197',
    'f062f6c1128e0e9d51a76747d9018849',
    'f39bf22d9a2f313425ee201932bac91a',
]


def remove_pen_marks(img):
    # Define elliptic kernel
    kernel5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # use cv2.inRange to mask pen marks (hardcoded for now)
    lower = np.array([0, 0, 0])
    upper = np.array([200, 255, 255])
    img_mask1 = cv2.inRange(img, lower, upper)

    # Use erosion and findContours to remove masked tissue (side effect of above)
    img_mask1 = cv2.erode(img_mask1, kernel5x5, iterations=4)
    img_mask2 = np.zeros(img_mask1.shape, dtype=np.uint8)
    contours, _ = cv2.findContours(img_mask1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y = contour[:, 0, 0], contour[:, 0, 1]
        w, h = x.max() - x.min(), y.max() - y.min()
        if w > 100 and h > 100:
            cv2.drawContours(img_mask2, [contour], 0, 1, -1)
    # expand the area of the pen marks
    img_mask2 = cv2.dilate(img_mask2, kernel5x5, iterations=3)
    img_mask2 = (1 - img_mask2)

    # Mask out pen marks from original image
    img = cv2.bitwise_and(img, img, mask=img_mask2)

    img[img == 0] = 255

    return img, img_mask1, img_mask2


class TileMaker:

    def __init__(self, size, number):
        self.size = size
        self.number = number

    def make_multistride(self, image, mask):
        # Pad only once
        image, mask = self.__pad(image, mask)
        s0, _ = self.__get_tiles(image, mask)

        # For strided grids, need to also remove on the right/bottom
        s1, _ = self.__get_tiles(image[self.size // 2:-self.size // 2, :],
                                 mask[self.size // 2:-self.size // 2, :])
        s2, _ = self.__get_tiles(image[:, self.size // 2:-self.size // 2],
                                 image[:, self.size // 2:-self.size // 2])
        s3, _ = self.__get_tiles(image[self.size // 2:-self.size // 2, self.size // 2:-self.size // 2],
                                 image[self.size // 2:-self.size // 2, self.size // 2:-self.size // 2])

        all_tiles = np.concatenate([s0, s1, s2, s3], axis=0)
        # Find the images with the most stuff (the most red):
        red_channel = all_tiles[:, :, :, 0]
        tissue = np.where((red_channel < 230) & (red_channel > 200), red_channel, 0)
        sorted_tiles = np.argsort(np.sum(tissue, axis=(1, 2)))[::-1]
        sorted_tiles = sorted_tiles[:self.number * 4]

        return all_tiles[sorted_tiles], _

    def __pad(self, image, mask):
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
        return image, mask

    def __get_tiles(self, image, mask):
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

        return image, mask

    def make(self, image, mask):
        image, mask = self.__pad(image, mask)
        image, mask = self.__get_tiles(image, mask)
        # Find the images with the most dark (epithelium) stuff
        red_channel = image[:, :, :, 0]
        tissue = np.where((red_channel < 230) & (red_channel > 200), red_channel, 0)
        sorted_tiles = np.argsort(np.sum(tissue, axis=(1, 2)))[::-1]
        sorted_tiles = sorted_tiles[:self.number]

        return image[sorted_tiles], mask[sorted_tiles]


if __name__ == "__main__":

    OUTPUT_IMG_PATH.mkdir(exist_ok=True, parents=True)
    OUTPUT_MASK_PATH.mkdir(exist_ok=True, parents=True)

    tile_maker = TileMaker(SIZE, NUM)

    img_list = list(TRAIN_PATH.glob('**/*.tiff'))
    # img_list.pop(5765)
    bad_images = []
    bad_masks = []
    image_stats = []
    files = []
    for i, img_fn in enumerate(img_list):

        img_id = img_fn.stem
        mask_fn = MASKS_TRAIN_PATH / (img_id + '_mask.tiff')

        try:
            col = skimage.io.MultiImage(str(img_fn))
            image = col[-LEVEL]
        except:
            bad_images.append(img_id)
            continue

        if img_id in pen_marked_images:
            image, _, _ = remove_pen_marks(image)

        if mask_fn.exists():

            try:
                mask = skimage.io.MultiImage(str(mask_fn))[-LEVEL]
            except:
                bad_masks.append(img_id)
                mask = np.zeros_like(image)

        else:
            mask = np.zeros_like(image)

        if STRIDE:
            image, mask = tile_maker.make_multistride(image, mask)
        else:
            image, mask = tile_maker.make(image, mask)
        sys.stdout.write(f'\r{i + 1}/{len(img_list)}')

        image_stats.append({'image_id': img_id, 'mean': image.mean(axis=(0, 1, 2)) / 255,
                            'mean_square': ((image / 255) ** 2).mean(axis=(0, 1, 2)),
                            'img_mean': (255 - image).mean()})

        for i, (tile_image, tile_mask) in enumerate(zip(image, mask)):
            a = (img_id + '_' + str(i) + '.png')
            b = (img_id + '_' + str(i) + '.png')
            files.append({'image_id': img_id, 'num': i, 'filename': a, 'maskname': b,
                          'value': (255-tile_image[:, :, 0]).mean()})
            skimage.io.imsave(OUTPUT_IMG_PATH / a, tile_image, check_contrast=False)
            skimage.io.imsave(OUTPUT_MASK_PATH / b, tile_mask, check_contrast=False)

    image_stats = pd.DataFrame(image_stats)
    df = pd.read_csv(CSV_PATH)
    df = pd.merge(df, image_stats, on='image_id', how='left')
    df[['image_id', 'img_mean']].to_csv(OUTPUT_BASE/f'img_mean_{SIZE}_{LEVEL}.csv', index=False)

    provider_stats = {}
    for provider in df['data_provider'].unique():
        mean = (df[df['data_provider'] == provider]['mean']).mean(0)
        std = np.sqrt((df[df['data_provider'] == provider]['mean_square']).mean(0) - mean ** 2)
        provider_stats[provider] = (mean, std)

    mean = (df['mean']).mean()
    std = np.sqrt((df['mean_square']).mean() - mean ** 2)
    provider_stats['all'] = (mean, std)

    with open(PICKLE_NAME, 'wb') as file:
        pickle.dump(provider_stats, file)

    pd.DataFrame(files).to_csv(OUTPUT_BASE/f'files_{SIZE}_{LEVEL}.csv', index=False)

    print(bad_images)
    print(bad_masks)
    print(provider_stats)

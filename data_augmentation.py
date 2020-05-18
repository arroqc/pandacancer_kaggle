import numpy as np
import torch
from albumentations import (HorizontalFlip, VerticalFlip, ShiftScaleRotate, Transpose, Compose, GaussNoise,
                            RandomBrightnessContrast, HueSaturationValue)


class AlbumentationTransform:
    def __init__(self, p=0.5):
        self.strong_aug = strong_aug(p)

    def __call__(self, image):
        image = np.array(image)
        return self.strong_aug(image=image)['image']


# def strong_aug(p=.5):
#     return OneOf([
#         RandomRotate90(),
#         Flip(),
#         ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=20, p=0.6, value=255),
#         RandomBrightnessContrast(p=0.1),
#         HueSaturationValue(hue_shift_limit=10, p=0.1),
#     ], p=p)


def strong_aug(p=.5):
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Transpose(p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
        GaussNoise(p=0.2),
        RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.2),
        HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, p=0.2)
        ], p=1)


class TilesRandomDuplicate:

    def __init__(self, p, num=2):
        self.p = p
        self.num = num

    def __call__(self, tiles):

        if np.random.rand() < self.p:
            # tiles is tensor N, C, H, W
            n, c, h, w, = tiles.shape
            idx_source = np.random.randint(0, n, self.num)
            idx_dest = np.random.randint(0, n, self.num)

            for idx_s, idx_d in zip(idx_source, idx_dest):
                tiles[idx_dest] = tiles[idx_source]

        return tiles


class TilesRandomRemove:

    def __init__(self, p, num=1):
        self.p = p
        self.num = num

    def __call__(self, tiles):

        if np.random.rand() < self.p:
            # tiles is tensor N, C, H, W
            n, c, h, w, = tiles.shape
            idx_remove = np.random.randint(0, n, self.num)

            for idx_r in zip(idx_remove):
                tiles[idx_r] = torch.zeros((c, h, w))

        return tiles


class TilesCompose:

    def __init__(self, list_transforms):
        self.transforms = list_transforms

    def __call__(self, tiles):
        for t in self.transforms:
            tiles = t(tiles)
        return tiles
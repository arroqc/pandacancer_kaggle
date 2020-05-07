import numpy as np
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)


class AlbumentationTransform:
    def __init__(self, p=0.5):
        self.strong_aug = strong_aug(p)

    def __call__(self, image):
        image = np.array(image)
        return self.strong_aug(image=image)['image']


def strong_aug(p=.5):
    return OneOf([
        RandomRotate90(),
        Flip(),
        ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=20, p=0.6, value=255),
        RandomBrightnessContrast(p=0.1),
        HueSaturationValue(hue_shift_limit=10, p=0.1),
    ], p=p)

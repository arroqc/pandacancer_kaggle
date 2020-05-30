import skimage.io
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import pickle
import argparse
import cv2


def trim_background(image):
    ## (1) Convert to gray, and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    ## (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    ## (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    ## (4) Crop and save it
    x,y,w,h = cv2.boundingRect(cnt)
    dst = image[y:y+h, x:x+w]
    return dst


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
    def __init__(self, size, number, scale):
        self.size = size
        self.number = number
        self.scale = scale

    def __pad(self, image, constant_values=255):
        h, w, c = image.shape
        horizontal_pad = 0 if (w % self.size) == 0 else self.size - (w % self.size)
        vertical_pad = 0 if (h % self.size) == 0 else self.size - (h % self.size)

        image = np.pad(image, pad_width=((vertical_pad // 2, vertical_pad - vertical_pad // 2),
                                         (horizontal_pad // 2, horizontal_pad - horizontal_pad // 2),
                                         (0, 0)),
                       mode='constant', constant_values=constant_values)  # Empty is white in this data
        return image

    def __call__(self, image, constant_values=255):

        image = self.__pad(image, constant_values=constant_values)
        h, w, c = image.shape
        image = image.reshape(h // self.size, self.size, w // self.size, self.size, c)
        image = image.swapaxes(1, 2).reshape(-1, self.size, self.size, c)

        if image.shape[0] < self.number:
            image = np.pad(image, pad_width=((0, self.number - image.shape[0]), (0, 0), (0, 0), (0, 0)),
                           mode='constant', constant_values=constant_values)

        sorted_tiles = np.argsort(np.sum(image, axis=(1, 2, 3)))
        sorted_tiles = sorted_tiles[:self.number]

        image = image[sorted_tiles]

        return image


def save_to_disk(tiles, img_id, prefix='', suffix=''):
    stats = []
    for i, tile_image in enumerate(tiles):
        filename = prefix + (img_id + '_' + str(i) + suffix + '.png')
        stats.append({'image_id': img_id, 'filename': filename, 'reverse_white_area': (255 - tile_image[:, :, 0]).mean()})
        skimage.io.imsave(OUTPUT_IMG_PATH / filename, tile_image, check_contrast=False)
    return stats


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), borderValue=(255, 255, 255))
    return rotated_mat


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default='G:/Datasets/panda', required=False)
    parser.add_argument("--out_dir", default='D:/Datasets/panda', required=False)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--num", required=True, type=int)
    parser.add_argument("--level", required=True, type=int)
    parser.add_argument("--scale", required=True, type=float)
    parser.add_argument("--sets", required=True, type=str)
    args = parser.parse_args()

    BASE_PATH = Path(args.base_dir)
    OUTPUT_BASE = Path(args.out_dir)

    SIZE = args.size
    NUM = args.num
    LEVEL = args.level
    SCALE = args.scale
    SETS = [int(num) for num in args.sets.split(',')]

    TRAIN_PATH = BASE_PATH / 'train_images/'
    MASKS_TRAIN_PATH = BASE_PATH / 'train_label_masks/'
    OUTPUT_IMG_PATH = OUTPUT_BASE / f'train_tiles_{SIZE}_{LEVEL}_{int(SCALE*10)}/imgs/'
    PICKLE_NAME = OUTPUT_BASE / f'stats_{SIZE}_{LEVEL}_{int(SCALE*10)}.pkl'
    CSV_PATH = BASE_PATH / 'train.csv'

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

    OUTPUT_IMG_PATH.mkdir(exist_ok=True, parents=True)
    df_train = pd.read_csv(CSV_PATH)

    img_list = list(TRAIN_PATH.glob('**/*.tiff'))
    image_stats = []
    tiles_stats = []

    tile_maker = TileMaker(SIZE, NUM, SCALE)
    for i, img_fn in enumerate(img_list):

        sys.stdout.write(f'\r{i + 1}/{len(img_list)}')

        img_id = img_fn.stem
        mask_fn = MASKS_TRAIN_PATH / (img_id + '_mask.tiff')

        col = skimage.io.MultiImage(str(img_fn))
        image = trim_background(col[-LEVEL])

        if SCALE != 1.0:
            h, w, _ = image.shape
            new_h, new_w = int(h * SCALE), int(w * SCALE)
            image = cv2.resize(image, (new_h, new_w))

        if img_id in pen_marked_images:
            image, _, _ = remove_pen_marks(image)

        if 0 in SETS:
            # Image stats
            tissue_mask = image[:, :, 0:1] < 240
            surface = tissue_mask.sum() / (SIZE ** 2)
            tissue_mask = np.repeat(tissue_mask, 3, axis=2)

            tissue = np.ma.masked_where(~tissue_mask, 1 - image/255)
            mean = tissue.mean(axis=(0, 1)).data
            std = ((tissue**2).mean(axis=(0, 1)).data - mean ** 2) ** 0.5

            image_stats.append({'image_id': img_id, 'surface': surface, 'mean': mean, 'std': std})

            # Normal images
            tiles = tile_maker(image)
            tiles_stats += save_to_disk(tiles, img_id, prefix='', suffix='')

        if 1 in SETS:
            # Stride right
            image_tr = image[:, SIZE // 2:-SIZE // 2]
            tiles = tile_maker(image_tr)
            tiles_stats += save_to_disk(tiles, img_id, prefix='', suffix='_1')

            # Stride down
            image_tr = image[SIZE // 2:-SIZE // 2, :]
            tiles = tile_maker(image_tr)
            tiles_stats += save_to_disk(tiles, img_id, prefix='', suffix='_2')

            # Stride right/down
            image_tr = image[SIZE // 2:-SIZE // 2, SIZE // 2:-SIZE // 2]
            tiles = tile_maker(image_tr)
            tiles_stats += save_to_disk(tiles, img_id, prefix='', suffix='_3')

        if 2 in SETS:
            # Rotate 15degrees
            image_tr = rotate_image(image, 15)
            tiles = tile_maker(image_tr)
            tiles_stats += save_to_disk(tiles, img_id, prefix='', suffix='_4')

            # Rotate -15degrees
            image_tr = rotate_image(image, -15)
            tiles = tile_maker(image_tr)
            tiles_stats += save_to_disk(tiles, img_id, prefix='', suffix='_5')

            # Rotate 15 degrees and stride right/down
            image_tr = rotate_image(image, 15)
            image_tr = image_tr[SIZE // 2:-SIZE // 2, SIZE // 2:-SIZE // 2]
            tiles = tile_maker(image_tr)
            tiles_stats += save_to_disk(tiles, img_id, prefix='', suffix='_6')

            # Rotate -15 degrees and stride right/down
            image_tr = rotate_image(image, -15)
            image_tr = image_tr[SIZE // 2:-SIZE // 2, SIZE // 2:-SIZE // 2]
            tiles = tile_maker(image_tr)
            tiles_stats += save_to_disk(tiles, img_id, prefix='', suffix='_7')

        if 3 in SETS:
            # Rotate 30degrees
            image_tr = rotate_image(image, 30)
            tiles = tile_maker(image_tr)
            tiles_stats += save_to_disk(tiles, img_id, prefix='', suffix='_8')

            # Rotate -30degrees
            image_tr = rotate_image(image, -30)
            tiles = tile_maker(image_tr)
            tiles_stats += save_to_disk(tiles, img_id, prefix='', suffix='_9')

            # Rotate 30 degrees and stride right/down
            image_tr = rotate_image(image, 30)
            image_tr = image_tr[SIZE // 2:-SIZE // 2, SIZE // 2:-SIZE // 2]
            tiles = tile_maker(image_tr)
            tiles_stats += save_to_disk(tiles, img_id, prefix='', suffix='_10')

            # Rotate -30 degrees and stride right/down
            image_tr = rotate_image(image, -30)
            image_tr = image_tr[SIZE // 2:-SIZE // 2, SIZE // 2:-SIZE // 2]
            tiles = tile_maker(image_tr)
            tiles_stats += save_to_disk(tiles, img_id, prefix='', suffix='_11')

        if 4 in SETS:
            # Rescale 0.1
            new_h, new_w = int(image.shape[0] * 1.1), int(image.shape[1] * 1.1)
            image_tr = cv2.resize(image, (new_w, new_h))
            tiles = tile_maker(image_tr)
            tiles_stats += save_to_disk(tiles, img_id, prefix='', suffix='_12')

            # Rescale -0.1
            new_h, new_w = int(image.shape[0] * 0.9), int(image.shape[1] * 0.9)
            image_tr = cv2.resize(image, (new_w, new_h))
            tiles = tile_maker(image_tr)
            tiles_stats += save_to_disk(tiles, img_id, prefix='', suffix='_13')

            # Rescale 0.1 and stride right/down
            new_h, new_w = int(image.shape[0] * 1.1), int(image.shape[1] * 1.1)
            image_tr = cv2.resize(image, (new_w, new_h))
            image_tr = image_tr[SIZE // 2:-SIZE // 2, SIZE // 2:-SIZE // 2]
            tiles = tile_maker(image_tr)
            tiles_stats += save_to_disk(tiles, img_id, prefix='', suffix='_14')

            # Rescale -0.1 and stride right/down
            new_h, new_w = int(image.shape[0] * 0.9), int(image.shape[1] * 0.9)
            image_tr = cv2.resize(image, (new_w, new_h))
            image_tr = image_tr[SIZE // 2:-SIZE // 2, SIZE // 2:-SIZE // 2]
            tiles = tile_maker(image_tr)
            tiles_stats += save_to_disk(tiles, img_id, prefix='', suffix='_15')

    if 5 in SETS:
        # Combination dataset ?
        comb_df = []
        for i in range(10000):

            pures = ['0+0', '3+3', '4+4', '5+5']
            providers = ['radboud', 'karolinska']
            rand_gleason = np.random.choice(pures)
            rand_provider = np.random.choice(providers)
            indexes = df_train[(df_train['gleason_score'] == rand_gleason) &
                               (df_train['data_provider'] == rand_provider)].index.values

            chosen_pair = np.random.choice(indexes, 2, replace=False)
            idx1, idx2 = chosen_pair
            meta1 = df_train.iloc[idx1]
            img_id1 = meta1['image_id']
            wsi = skimage.io.MultiImage(str(TRAIN_PATH / (img_id1 + '.tiff')))
            img1 = trim_background(wsi[-LEVEL])
            meta2 = df_train.iloc[idx2]
            img_id2 = meta2['image_id']
            wsi = skimage.io.MultiImage(str(TRAIN_PATH / (img_id2 + '.tiff')))
            img2 = trim_background(wsi[-LEVEL])
            h1, w1, _ = img1.shape
            h2, w2, _ = img2.shape

            if w1 > w2:
                pad2_w = (w1 - w2)
                pad1_w = 0
            else:
                pad2_w = 0
                pad1_w = (w2 - w1)
            if h1 > h2:
                pad2_h = (h1 - h2)
                pad1_h = 0
            else:
                pad2_h = 0
                pad1_h = (h2 - h1)

            img1 = np.pad(img1, ((pad1_h // 2, pad1_h - pad1_h // 2), (pad1_w // 2, pad1_w - pad1_w // 2), (0, 0)),
                          constant_values=255)
            img2 = np.pad(img2, ((pad2_h // 2, pad2_h - pad2_h // 2), (pad2_w // 2, pad2_w - pad2_w // 2), (0, 0)),
                          constant_values=255)
            img_comb = np.concatenate([img1, img2], axis=1)
            tiles = tile_maker(img_comb)
            tiles_stats += save_to_disk(tiles, img_id1 + '_' + img_id2, prefix='', suffix='_comb')

            comb_df.append({'image_id': i, 'idx1': idx1,
                            'idx2': idx2, 'image_id1': meta1['image_id'], 'image_id2': meta2['image_id'],
                            'isup': meta1['isup_grade'], 'data_provider': meta1['data_provider'], 'gleason_score': meta1['gleason_score']})
        comb_df = pd.DataFrame(comb_df)
        comb_df.to_csv(OUTPUT_BASE / f'comb_{SIZE}_{LEVEL}.csv', index=False)

    if 0 in SETS:
        image_stats = pd.DataFrame(image_stats)
        image_stats.to_csv(OUTPUT_BASE/f'imagestats_{SIZE}_{LEVEL}.csv', index=False)
    tiles_stats = pd.DataFrame(tiles_stats)
    tiles_stats.to_csv(OUTPUT_BASE/f'tilesstats_{SIZE}_{LEVEL}.csv', index=False)
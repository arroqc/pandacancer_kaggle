import numpy as np
import torch.utils.data as tdata
from pathlib import Path
from PIL import Image
import torch

suspicious_list = [
'3790f55cad63053e956fb73027179707',
'4a2ca53f240932e46eaf8959cb3f490a',
'3790f55cad63053e956fb73027179707',
'e4215cfc8c41ec04a55431cc413688a9',
'aaa5732cd49bffddf0d2b7d36fbb0a83',
'3790f55cad63053e956fb73027179707',
'4a2ca53f240932e46eaf8959cb3f490a',
'aaa5732cd49bffddf0d2b7d36fbb0a83',
'e4215cfc8c41ec04a55431cc413688a9',
'004dd32d9cd167d9cc31c13b704498af',
'00d8a8c04886379e266406fdeff81c45',
'014006841b9807edc0ff277c4ab29b91',
'01dfcde514052a6dc35ea4407f41d6e1',
'053a397c936bffc98e62367a81d6c905',
'0e5806abc1cf909123d584e504dd9bf9',
'0e62a4cba998a03d20295e07ebc30958',
'0f914d1f044c187e6a5be7e996d877a9',
'102aaf6f71973e1fe82551918255ba58',
'102c1ec848392aabff3fb4df2aa3598b',
'1163571b3edf666ab3a1dd30686e7530',
'158754df49e00760f8e4659a05e7cc0c',
'18566c8e26d5b14f599fc92acec008e0',
'1a60c75983db9d7a7d8acada5151700e',
'1f722fff98986499a0aea339cd05a75d',
'268dcbd78318c0108974c5f99e1dca78',
'294c41defe507592692b93bb10c33123',
'2a4acf0e2db5c6fab745a163af60259b',
'2ce127e7731155d1373bfbd0abdc368b',
'2e4281fa0f1f749d052579c9855476cf',
'3b994de767ea6fb282fe774f88f3fe3d',
'3c685ed0f92e30f6ffeb5cea7cef5994',
'3d459935dae8bbeb5bda9cfc40e4b3ab',
'3d667d7f5829b3d76b3b0cc52bd71719',
'3d7a6ed3325520d2325441bb3ede70ea',
'4031bad3ffbcd79f42330f3853bec9b2',
'44f60ac5d5d82e5c72d077e50701863d',
'49f0eacee68196c3e5b997ee6d82214e',
'501403e5c13c7b86bc9312d087a6e490',
'507bb5873324858d94e7ea6bd043fa66',
'50fb0ac6d41c214effff1c9cd462dd30',
'527d8fc0ea1ef55212f4f2ffd6c4ad68',
'5752b761672c420fc1df9cbaf860bcd6',
'5af447057c0eb2ca945c12c398f670f9',
'627aeb9327a85228458ad20ecd8d77c3',
'65b59d73dd788168afab76694d39529c',
'6ab7bc76941454c4ec71eb754f066f30',
'77e8a9d9718a78c7602a4cd5d1f98da5',
'7dd4b4c937f8c0ceb6a30e25391f45bf',
'7e7e6dac3271a6cad2744ae393328a36',
'821b7c89e5f5879c21cefcdeb0df1657',
'84da53ac6f8828a494a0a8c52a6cd158',
'896728c112dfe5f55662f5f17656e076',
'8a308bfd3df0d8bda517873600971a84',
'8be88cbd606502cf980f721d57c1c794',
'8d9e28e97edc6598da7df0b5b7c098d6',
'93f366029ac746d84ea2aea80cc998e8',
'9665b32251dfc1c438787d36e8b66dd0',
'9a9ca5c457ad34e83f7e83002c61550f',
'9c24ae764d87098d03faccc2e2a579d0',
'9d14c42db92e5ba66f97630b55e1724f',
'a01a079113eb9c126a407defe6571315',
'aa082e66a044358b7222c7e2fb499a2b',
'abefa7fbef36b2bcc0c67eebb672e820',
'b10aafff8d0a4c426090fc772b6075f0',
'b1a8dc23435d6fea4a7847fcf02f9374',
'b31da8462061d770024e293ca92e0086',
'b53d843700d9665439ec8791e4024d70',
'b5925cdedf4c9c40b4249a39bfbbe35a',
'b64b0fc3bbfde95fb265773a61962813',
'b65538bb9960f77e666ad4a6dcb6a77e',
'b9402571270ab62c1e0a690fd33fc9e6',
'bae0ff238ba5e0f675f06a72a383bcd6',
'bf282897514555144100efbee376bce1',
'c0852776d3142ce132a90c6142bd7146',
'c1b2ec5ced6e0232ca1cff1c559234f2',
'c2f90bbc5200609817a0745279582e72',
'c64c12966d164cd0bcc6bfef4ca95500',
'c867630b14137ca898056cb4eefe1696',
'cbad739adf8a8bf6803b77f81946f8a4',
'cdb7663719497428b4d9243b76da3ace',
'ce07aae37ab28e6276bcab332a9706c9',
'cfaec8e16eae41da6abd659145a865f7',
'd08bd91bd82152478e446b6c0aca6d4a',
'd14168b713f3ba30243a69837a001115',
'd3b6c2650c9f201f3ff4ad20faafc23a',
'd6b1c8ca6037b5ddca5d2086975a643b',
'eda25d74c2ca92594de5aab4cc100f5c',
'ee5e98f996430e0ba9152fafb562596e',
'eec44ee55b0dfc4a00a728d459a6c5bb',
'ef65ea2c6fcec3156ffb0034135f8d55',
'f1a7f32a0a9ed7c50796a10e150df72b',
'fe79209ab178c89a9be62bc05b63f083',
'ffc70bf605de30aaa936533397a29d9c',]


class TileDataset(tdata.Dataset):
    def __init__(self, img_path, dataframe, num_tiles, suffix,
                 transform=None,
                 target='class',
                 return_stitched=True,
                 rand=False,
                 use_suspicious=True,
                 tile_stats=None):

        self.suffix = suffix
        self.img_path = Path(img_path)
        self.target = target

        if not use_suspicious:
            dataframe = dataframe[~dataframe['image_id'].isin(suspicious_list)]

        self.df = dataframe
        self.num_tiles = num_tiles
        self.img_list = self.df['image_id'].values
        self.transform = transform
        self.return_stitched = return_stitched

        if self.suffix != '' and tile_stats is not None:
            if len(self.suffix) == 2:
                tile_stats = tile_stats[tile_stats['filename'].str[-6:-4] == suffix]
            if len(self.suffix) == 3:
                tile_stats = tile_stats[tile_stats['filename'].str[-7:-4] == suffix]
        if self.suffix == '' and tile_stats is not None:
            tile_stats = tile_stats[tile_stats['filename'].str.len().isin([38, 39])]

        self.tile_stats = tile_stats
        self.rand = rand

    def __getitem__(self, idx):
        img_id = self.img_list[idx]

        if self.rand:
            subdf = self.tile_stats[self.tile_stats['image_id'] == img_id]
            sample = subdf.sample(self.num_tiles,
                                  weights=subdf['reverse_white_area'] + 1e-6, replace=False).sort_values(
                                  by=['reverse_white_area'], ascending=False)
            file_list = sample['filename'].values
        else:
            file_list = [img_id + '_' + str(i) + self.suffix + '.png' for i in range(0, self.num_tiles)]

        tiles_paths = [str(self.img_path/fn) for fn in file_list]

        tiles = [np.array(Image.open(tile)) for tile in tiles_paths]
        metadata = self.df.iloc[idx]

        if self.return_stitched:
            images = self.make_square(tiles)
        else:
            images = self.make_bag(tiles)

        isup = metadata['isup_grade']
        if self.target == 'bin':
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

        elif self.target == 'one_hot':
            if isup == 0:
                isup = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32)
            elif isup == 1:
                isup = torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float32)
            elif isup == 2:
                isup = torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float32)
            elif isup == 3:
                isup = torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.float32)
            elif isup == 4:
                isup = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float32)
            elif isup == 5:
                isup = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32)
        elif self.target == 'class':
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

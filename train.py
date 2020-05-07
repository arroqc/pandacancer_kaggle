import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchvision.transforms as transforms
import torchvision.models as models
import PIL.Image as Image
from pathlib import Path
import pandas as pd
import numpy as np
import random
import os
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import pytorch_lightning as pl
from contribs.ranger import Ranger
from contribs.mish import Mish
from utils import dict_to_args


class TileDataset(tdata.Dataset):

    def __init__(self, img_path, dataframe, transform=None, normalize_stats=None):

        self.img_path = Path(img_path)
        self.df = dataframe
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

        tiles = self.img_path.glob('**/' + img_id + '*.png')
        metadata = self.df.iloc[idx]
        image_tiles = []
        for tile in tiles:
            image = Image.open(tile)

            if self.transform is not None:
                image = self.transform(image)

            if self.normalize_stats is not None:
                provider = metadata['data_provider']
                image = self.normalize_stats[provider](image)
                image_tiles.append(image)

        image_tiles = torch.stack(image_tiles, dim=0)

        return {'image': image_tiles, 'provider': metadata['data_provider'],
                'isup': metadata['isup_grade'], 'gleason': metadata['gleason_score']}

    def __len__(self):
        return len(self.img_list)


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, x):
        avg_x = self.avg(x)
        max_x = self.max(x)
        return torch.cat([avg_x, max_x], dim=1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Model(nn.Module):

    def __init__(self, c_feature=2048, c_out=6, pretrained=True, **kwargs):
        super().__init__()
        self.feature_extractor = nn.Sequential(*list(models.resnet50(pretrained=pretrained).children())[:-2])
        print(c_out, c_feature)
        self.head = nn.Sequential(AdaptiveConcatPool2d(),
                                  Flatten(),
                                  nn.Linear(c_feature * 2, 512),
                                  Mish(),
                                  nn.BatchNorm1d(512),
                                  nn.Dropout(0.5),
                                  nn.Linear(512, c_out))

    def forward(self, x):
        h = x.view(-1, 3, 128, 128)
        h = self.feature_extractor(h)
        bn, c, height, width = h.shape
        h = h.view(-1, 16, c, height, width).permute(0, 2, 1, 3, 4).contiguous().view(-1, c, height * 16, width)
        h = self.head(h)
        return h


class LightModel(pl.LightningModule):

    def __init__(self, train_idx, val_idx, provider_stats, hparams):
        super().__init__()
        self.train_idx = train_idx
        self.val_idx = val_idx

        self.model = Model(c_feature=2048,
                           c_out=6,
                           pretrained=hparams.pretrained)
        self.provider_stats = provider_stats
        self.hparams = hparams

    def forward(self, batch):
        return self.model(batch['image'])

    def prepare_data(self):
        from data_augmentation import AlbumentationTransform
        transform_train = transforms.Compose([AlbumentationTransform(0.5),
                                              transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
        self.trainset = TileDataset(TRAIN_PATH, df_train.iloc[self.train_idx], transform=transform_train,
                                    normalize_stats=self.provider_stats)
        self.valset = TileDataset(TRAIN_PATH, df_train.iloc[self.val_idx], transform=transform_test,
                                  normalize_stats=self.provider_stats)

    def train_dataloader(self):
        train_dl = tdata.DataLoader(self.trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
        return train_dl

    def val_dataloader(self):
        val_dl = tdata.DataLoader(self.valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
        return [val_dl]

    def cross_entropy_loss(self, logits, gt):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, gt)

    def configure_optimizers(self):
        optimizer = Ranger(self.model.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.cross_entropy_loss(logits, batch['isup']).unsqueeze(0)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.cross_entropy_loss(logits, batch['isup']).unsqueeze(0)
        preds = logits.argmax(1)
        return {'val_loss': loss, 'preds': preds, 'gt': batch['isup']}

    def validation_end(self, outputs):
        avg_loss = torch.cat([out['val_loss'] for out in outputs], dim=0).mean()
        preds = torch.cat([out['preds'] for out in outputs], dim=0)
        gt = torch.cat([out['gt'] for out in outputs], dim=0)
        preds = preds.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
        kappa = cohen_kappa_score(preds, gt, weights='quadratic')
        tensorboard_logs = {'val_loss': avg_loss, 'kappa': kappa}

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


if __name__ == '__main__':
    TRAIN_PATH = 'D:/Datasets/panda/train_tiles/imgs/'
    CSV_PATH = 'G:/Datasets/panda/train.csv'
    SEED = 34
    BATCH_SIZE = 8
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    df_train = pd.read_csv(CSV_PATH)
    df_train = df_train[~(df_train['image_id'].isin(['8d90013d52788c1e2f5f47ad80e65d48']))]
    kfold = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    splits = kfold.split(df_train, df_train['isup_grade'])
    train_idx, val_idx = next(splits)

    with open('./stats.pkl', 'rb') as file:
        provider_stats = pickle.load(file)

    hparams = {'lr': 5e-4,
               'pretrained': True}
    model = LightModel(train_idx, val_idx, provider_stats, dict_to_args(hparams))
    trainer = pl.Trainer(gpus=[0], max_nb_epochs=10, auto_lr_find=False)
    # lr_finder = trainer.lr_find(model)
    # fig = lr_finder.plot(suggest=True)
    # fig.savefig('lr_plot.png')
    trainer.fit(model)

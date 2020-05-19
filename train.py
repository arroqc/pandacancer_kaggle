import pandas as pd
import numpy as np
import random
import os
import pickle
import datetime
import argparse

import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchvision.transforms as transforms
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

from contribs.torch_utils import split_weights, FlatCosineAnnealingLR
from contribs.fancy_optimizers import Over9000
from contribs.kappa_rounder import OptimizedRounder_v2
from datasets import TileDataset
from modules import Model
from utils import dict_to_args
from data_augmentation import AlbumentationTransform, TilesCompose, TilesRandomDuplicate, TilesRandomRemove


parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", default='D:/Datasets/panda', required=False)
args = parser.parse_args()
ROOT_PATH = args.root_dir


class LightModel(pl.LightningModule):

    def __init__(self, train_idx, val_idx, provider_stats, hparams):
        super().__init__()
        self.train_idx = train_idx
        self.val_idx = val_idx

        if hparams.task == 'regression':
            c_out = 1
        else:
            c_out = 6

        self.model = Model(c_out=c_out,
                           n_tiles=hparams.n_tiles,
                           tile_size=hparams.tile_size,
                           backbone=hparams.backbone,
                           head=hparams.head)

        self.provider_stats = provider_stats
        self.hparams = hparams
        self.opt = None
        self.trainset = None
        self.valset = None

    def forward(self, batch):
        return self.model(batch['image'])

    def prepare_data(self):
        transform_train = transforms.Compose([AlbumentationTransform(0.5),
                                              transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])

        tiles_transform = None
        if self.hparams.tiles_data_augmentation:
            tiles_transform = TilesCompose([TilesRandomRemove(p=0.7, num=4),
                                            TilesRandomDuplicate(p=0.7, num=4)])

        self.trainset = TileDataset(TRAIN_PATH, df_train.iloc[self.train_idx], num_tiles=self.hparams.n_tiles, transform=transform_train,
                                    normalize_stats=self.provider_stats, tiles_transform=tiles_transform)
        self.valset = TileDataset(TRAIN_PATH, df_train.iloc[self.val_idx], num_tiles=self.hparams.n_tiles, transform=transform_test,
                                  normalize_stats=self.provider_stats)

    def train_dataloader(self):
        train_dl = tdata.DataLoader(self.trainset, batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=min(6, os.cpu_count()))
        return train_dl

    def val_dataloader(self):
        val_dl = tdata.DataLoader(self.valset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=min(6, os.cpu_count()))
        return [val_dl]

    def cross_entropy_loss(self, logits, gt):
        if self.hparams.task == 'regression':
            if self.hparams.reg_loss == 'mse':
                loss_fn = nn.MSELoss()
            elif self.hparams.reg_loss == 'smooth_l1':
                loss_fn = nn.SmoothL1Loss()
            gt = gt.unsqueeze(1).float()
        else:
            loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, gt)

    def configure_optimizers(self):

        if self.hparams.weight_decay:
            params_backbone = split_weights(self.model.feature_extractor)
            params_backbone[0]['lr'] = self.hparams.lr_backbone
            params_backbone[1]['lr'] = self.hparams.lr_backbone
            params_head = split_weights(self.model.head)
            params_head[0]['lr'] = self.hparams.lr_head
            params_head[1]['lr'] = self.hparams.lr_head
            params = params_backbone + params_head
        else:
            params_backbone = self.model.feature_extractor.parameters()
            params_head = self.model.head.parameters()
            params = [dict(params=params_backbone, lr=self.hparams.lr_backbone),
                      dict(params=params_head, lr=self.hparams.lr_head)]

        optimizer = Over9000(params, weight_decay=3e-6)
        scheduler = FlatCosineAnnealingLR(optimizer, max_iter=EPOCHS, step_size=self.hparams.step_size)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.cross_entropy_loss(logits, batch['isup']).unsqueeze(0)
        if self.hparams.task == 'regression':
            preds = logits.squeeze(1)
        else:
            preds = logits.argmax(1)
        return {'loss': loss, 'preds': preds, 'gt': batch['isup'], 'log': {'train_loss': loss}}

    def training_epoch_end(self, outputs):
        preds = torch.cat([out['preds'] for out in outputs], dim=0)
        gt = torch.cat([out['gt'] for out in outputs], dim=0)
        preds = preds.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()

        if self.hparams.task == 'regression' and self.hparams.opt_fit == 'train':
            self.opt = OptimizedRounder_v2(6)
            self.opt.fit(preds, gt)

        return {}

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.cross_entropy_loss(logits, batch['isup']).unsqueeze(0)
        if self.hparams.task == 'regression':
            preds = logits.squeeze(1)
        else:
            preds = logits.argmax(1)
        return {'val_loss': loss, 'preds': preds, 'gt': batch['isup']}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.cat([out['val_loss'] for out in outputs], dim=0).mean()
        preds = torch.cat([out['preds'] for out in outputs], dim=0)
        gt = torch.cat([out['gt'] for out in outputs], dim=0)
        preds = preds.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()

        if self.hparams.task == 'regression':
            if self.hparams.use_opt:
                if self.opt is None or self.hparams.opt_fit == 'val':
                    self.opt = OptimizedRounder_v2(6)
                    self.opt.fit(preds, gt)
                preds = self.opt.predict(preds)
            else:
                preds = np.round(preds)

        kappa = cohen_kappa_score(preds, gt, weights='quadratic')
        tensorboard_logs = {'val_loss': avg_loss, 'kappa': kappa}
        print(f'Epoch {self.current_epoch}: {avg_loss:.2f}, kappa: {kappa:.4f}')

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


if __name__ == '__main__':

    EPOCHS = 30
    SEED = 33
    BATCH_SIZE = 16

    hparams = {'backbone': 'resnext50_semi',
               'head': 'basic',
               'lr_head': 1e-3,
               'lr_backbone': 1e-4,
               'n_tiles': 12,
               'level': 1,
               'tile_size': 128,
               'task': 'regression',
               'weight_decay': False,
               'pretrained': True,
               'use_opt': True,
               'opt_fit': 'train',
               'tiles_data_augmentation': False,
               'reg_loss': 'mse',
               'step_size': 8/EPOCHS}

    LEVEL = hparams['level']
    SIZE = hparams['tile_size']
    TRAIN_PATH = ROOT_PATH + f'/train_tiles_{SIZE}_{LEVEL}/imgs/'
    CSV_PATH = ROOT_PATH + '/train.csv'
    NAME = 'resnext50'
    OUTPUT_DIR = './lightning_logs'
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
    with open(f'{ROOT_PATH}/stats_{SIZE}_{LEVEL}.pkl', 'rb') as file:
        provider_stats = pickle.load(file)
    # values = pd.read_csv(f'{ROOT_PATH}/files_{SIZE}_{LEVEL}.csv')

    date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f'Fold {fold + 1}')
        tb_logger = pl.loggers.TensorBoardLogger(save_dir=OUTPUT_DIR,
                                                 name=f'{NAME}' + '-' + date,
                                                 version=f'fold_{fold + 1}')

        checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=tb_logger.log_dir + "/{epoch:02d}-{kappa:.4f}",
                                                           monitor='kappa', mode='max')

        model = LightModel(train_idx, val_idx, provider_stats, dict_to_args(hparams))
        trainer = pl.Trainer(gpus=[0], max_nb_epochs=EPOCHS, auto_lr_find=False,
                             gradient_clip_val=1,
                             logger=tb_logger,
                             accumulate_grad_batches=1,              # BatchNorm ?
                             checkpoint_callback=checkpoint_callback
                             )
        # lr_finder = trainer.lr_find(model)
        # fig = lr_finder.plot(suggest=True)
        # fig.savefig('lr_plot.png')
        trainer.fit(model)

        if hparams['use_opt']:
            print(model.opt.coefficients())
            with open(f'{OUTPUT_DIR}/{NAME}-{date}/fold{fold + 1}_coef.pkl', 'wb') as file:
                pickle.dump(file=file, obj=list(np.sort(model.opt.coefficients())))

        # Todo: One fold training
        break

# Tests to do:
# L1Smooth (small improvement)
# Gradient accumulation
# Test each options
# Level 1 images > Increase size and/or number
# Longer training
# RandomTileDataset
# Max/avg pool per tile then self attention.
# Use both level 1 and level 2

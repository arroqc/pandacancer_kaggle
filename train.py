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
from sklearn.metrics import confusion_matrix

from contribs.torch_utils import split_weights, FlatCosineAnnealingLR
from contribs.fancy_optimizers import Over9000
from contribs.kappa_rounder import OptimizedRounder_v2
from datasets import TileDataset
from modules import Model
from utils import dict_to_args
from data_augmentation import AlbumentationTransform, TilesCompose, TilesRandomDuplicate, TilesRandomRemove
import seaborn as sn
import matplotlib.pyplot as plt
import io
from PIL import Image


def convert_to_image(cm):
    df_cm = pd.DataFrame(cm, index=[i for i in "012345"],
                         columns=[i for i in "012345"])
    plt.figure(figsize=(10, 7))
    sns_plot = sn.heatmap(df_cm, annot=True)
    buf = io.BytesIO()
    sns_plot.get_figure().savefig(buf)
    cm_image = np.array(Image.open(buf).resize((512, 512)))[:, :, :3]
    return cm_image


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
                                    num_workers=self.hparams.num_workers)
        return train_dl

    def val_dataloader(self):
        val_dl = tdata.DataLoader(self.valset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=self.hparams.num_workers)
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

        if self.hparams.opt_algo == 'over9000':
            optimizer = Over9000(params, weight_decay=3e-6)
            scheduler = FlatCosineAnnealingLR(optimizer, max_iter=EPOCHS, step_size=self.hparams.step_size)
            return [optimizer], [scheduler]

        elif self.hparams.opt_algo == 'adam':
            optimizer = torch.optim.Adam(params, weight_decay=3e-6)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
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
        return {'val_loss': loss, 'preds': preds, 'gt': batch['isup'], 'provider': batch['provider']}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.cat([out['val_loss'] for out in outputs], dim=0).mean()
        preds = torch.cat([out['preds'] for out in outputs], dim=0)
        gt = torch.cat([out['gt'] for out in outputs], dim=0)
        provider = np.concatenate([out['provider'] for out in outputs], axis=0)
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
        cm = confusion_matrix(gt, preds)
        print('CM')
        print(cm)
        cm_radboud = confusion_matrix(gt[provider == 'radboud'], preds[provider == 'radboud'])
        cm_karolinska = confusion_matrix(gt[provider == 'karolinska'], preds[provider == 'karolinska'])

        kappa_radboud = cohen_kappa_score(gt[provider == 'radboud'],
                                          preds[provider == 'radboud'],
                                          weights='quadratic')
        kappa_karolinska = cohen_kappa_score(gt[provider == 'karolinska'],
                                             preds[provider == 'karolinska'],
                                             weights='quadratic')
        cm_image = convert_to_image(cm)
        self.logger.experiment.add_image('CM', cm_image, self.global_step, dataformats='HWC')
        cm_image = convert_to_image(cm_radboud)
        self.logger.experiment.add_image('CM Radboud', cm_image, self.global_step, dataformats='HWC')
        cm_image = convert_to_image(cm_karolinska)
        self.logger.experiment.add_image('CM Karolinska', cm_image, self.global_step, dataformats='HWC')
        print(f'Epoch {self.current_epoch}: {avg_loss:.2f}, kappa: {kappa:.4f}')
        print('CM radboud')
        print(cm_radboud)
        print('kappa radboud:', kappa_radboud)
        print('CM karolinska')
        print(cm_karolinska)
        print('kappa karolinska:', kappa_karolinska)

        kappa = torch.tensor(kappa)
        tensorboard_logs = {'val_loss': avg_loss, 'kappa': kappa, 'kappa_radboud': kappa_radboud,
                            'kappa_karolinska': kappa_karolinska}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


if __name__ == '__main__':

    EPOCHS = 30
    SEED = 33
    BATCH_SIZE = 6
    PRECISION = 32
    NUM_WORKERS = 6

    hparams = {'backbone': 'resnext50_semi',
               'head': 'basic',
               'lr_head': 1e-3,
               'lr_backbone': 1e-4,
               'n_tiles': 32,
               'level': 2,
               'tile_size': 128,
               'task': 'classification',
               'weight_decay': False,
               'pretrained': True,
               'use_opt': True,
               'opt_fit': 'val',
               'tiles_data_augmentation': False,
               'reg_loss': 'mse',
               'opt_algo': 'over9000',
               'num_workers': NUM_WORKERS,
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
                             accumulate_grad_batches=2,              # BatchNorm ?
                             checkpoint_callback=checkpoint_callback,
                             nb_sanity_val_steps=0,
                             precision=PRECISION
                             )
        # lr_finder = trainer.lr_find(model)
        # fig = lr_finder.plot(suggest=True)
        # fig.savefig('lr_plot.png')
        trainer.fit(model)

        # Fold predictions
        from pathlib import Path
        print('Load best checkpoint')
        ckpt = list(Path(tb_logger.log_dir).glob('*.ckpt'))[0]
        model.load_from_checkpoint(str(ckpt), val_idx=val_idx, provider_stats=provider_stats,
                                   hparams=dict_to_args(hparams))
        torch_model = model.model.eval().to('cuda')
        preds = []
        gt = []
        with torch.no_grad():
            for batch in model.val_dataloader()[0]:
                image = batch['image'].to('cuda')
                pred = torch_model(image)
                gt.append(batch['isup'])
                preds.append(pred)
        preds = torch.cat(preds, dim=0).squeeze(1).detach().cpu().numpy()
        gt = torch.cat(gt, dim=0).detach().cpu().numpy()
        if hparams['task'] == 'classification':
            pd.DataFrame({'val_idx': val_idx, 'preds0': preds[:, 0], 'preds1': preds[:, 1], 'preds2': preds[:, 2],
                          'preds3': preds[:, 3], 'preds4': preds[:, 4], 'preds5': preds[:, 5], 'gt': gt}).to_csv(
                f'{OUTPUT_DIR}/{NAME}-{date}/fold{fold + 1}_preds.csv', index=False)
        else:
            pd.DataFrame({'val_idx': val_idx, 'preds': preds, 'gt': gt}).to_csv(
                f'{OUTPUT_DIR}/{NAME}-{date}/fold{fold + 1}_preds.csv', index=False)

        if hparams['use_opt'] and hparams['opt_fit'] == 'val' and hparams['task'] == 'regression':
            opt = OptimizedRounder_v2(6)
            opt.fit(preds, gt)
            print(opt.coefficients())
            with open(f'{OUTPUT_DIR}/{NAME}-{date}/fold{fold + 1}_coef.pkl', 'wb') as file:
                pickle.dump(file=file, obj=list(np.sort(opt.coefficients())))

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

# Multi head attention pool.
# Tree method on the pooled representation

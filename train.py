import pandas as pd
import numpy as np
import random
import os
import pickle
import datetime

import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchvision.transforms as transforms
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

from contribs.utils import split_weights, FlatCosineAnnealingLR
from contribs.over9000 import Over9000
from contribs.rounder import OptimizedRounder_v2
from datasets import TileDataset
from modules import Model
from utils import dict_to_args


class LightModel(pl.LightningModule):

    def __init__(self, train_idx, val_idx, provider_stats, hparams):
        super().__init__()
        self.train_idx = train_idx
        self.val_idx = val_idx

        if hparams.task == 'regression':
            self.model = Model(c_out=1,
                               n_tiles=hparams.n_tiles,
                               pretrained=hparams.pretrained)
        else:
            self.model = Model(c_out=6,
                               n_tiles=hparams.n_tiles,
                               pretrained=hparams.pretrained)
        self.provider_stats = provider_stats
        self.hparams = hparams
        self.opt = None
        self.trainset = None
        self.valset = None

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
        train_dl = tdata.DataLoader(self.trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())
        return train_dl

    def val_dataloader(self):
        val_dl = tdata.DataLoader(self.valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())
        return [val_dl]

    def cross_entropy_loss(self, logits, gt):
        if self.hparams.task == 'regression':
            loss_fn = nn.MSELoss()
            gt = gt.unsqueeze(1).float()
        else:
            loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, gt)

    def configure_optimizers(self):
        if self.hparams.weight_decay:
            params = split_weights(self.model)
        else:
            params = self.model.parameters()
        optimizer = Over9000(params, lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = FlatCosineAnnealingLR(optimizer, max_iter=EPOCHS)
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

        if self.hparams.task == 'regression':
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
            if self.opt is None:
                self.opt = OptimizedRounder_v2(6)
                self.opt.fit(preds, gt)
            preds = self.opt.predict(preds)

        kappa = cohen_kappa_score(preds, gt, weights='quadratic')
        tensorboard_logs = {'val_loss': avg_loss, 'kappa': kappa}

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


if __name__ == '__main__':
    TRAIN_PATH = 'D:/Datasets/panda/train_tiles/imgs/'
    CSV_PATH = 'G:/Datasets/panda/train.csv'
    SEED = 34
    BATCH_SIZE = 8
    EPOCHS = 20
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
    with open('./stats.pkl', 'rb') as file:
        provider_stats = pickle.load(file)
    hparams = {'lr': 1e-4,
               'n_tiles': 12,
               'task': 'regression',
               'weight_decay': True,
               'pretrained': True}
    date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    for fold, (train_idx, val_idx) in enumerate(splits):

        tb_logger = pl.loggers.TensorBoardLogger(save_dir=OUTPUT_DIR,
                                                 name=f'{NAME}' + '-' + date,
                                                 version=f'fold_{fold+1}')

        checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=tb_logger.log_dir + "/{epoch:02d}-{kappa:.4f}",
                                                           monitor='kappa', mode='max')

        model = LightModel(train_idx, val_idx, provider_stats, dict_to_args(hparams))
        trainer = pl.Trainer(gpus=[0], max_nb_epochs=EPOCHS, auto_lr_find=False,
                             logger=tb_logger,
                             checkpoint_callback=checkpoint_callback
                             )
        # lr_finder = trainer.lr_find(model)
        # fig = lr_finder.plot(suggest=True)
        # fig.savefig('lr_plot.png')
        trainer.fit(model)

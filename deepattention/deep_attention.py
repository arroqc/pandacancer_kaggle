import torch.utils.data as tdata
import pytorch_lightning as pl
import skimage.io
import cv2
import numpy as np
from pathlib import Path
from deepattention.ats_layer import AttentionModelColonCancer, FeatureModelColonCancer, ClassificationHead, ATSModel
from deepattention.ats_layer import MultinomialEntropy
from torchvision import transforms
import torch.nn as nn
import torch
import os
from contribs.fancy_optimizers import Over9000
from contribs.torch_utils import FlatCosineAnnealingLR
from sklearn.metrics import cohen_kappa_score


class TiffDataset(tdata.Dataset):

    def __init__(self, img_path, df, transform):
        self.tiffpath = Path(img_path)
        self.df = df
        self.imglist = self.df['image_id'].values
        self.transform = transform

    def __getitem__(self, idx):
        img_fn = self.tiffpath/(self.imglist[idx] + '.tiff')

        collec = skimage.io.MultiImage(str(img_fn))
        rotated = False
        high_res = collec[-2]

        h, w, c = high_res.shape
        if h > w:
            rotated = True
            high_res = cv2.rotate(high_res, cv2.ROTATE_90_CLOCKWISE)
            h, w, c = high_res.shape

        pad = w - h
        high_res = np.pad(high_res, ((pad//2 * 16, pad//2 * 16), (0, 0), (0, 0)), constant_values=255)
        low_res = cv2.resize(high_res, (224, 224))
        self.transform(high_res)
        high_res = 1 - high_res
        high_res = transforms.Normalize([1.0 - 0.90949707, 1.0 - 0.8188697, 1.0 - 0.87795304],
                                        [0.36357649, 0.49984502, 0.40477625])(high_res)
        metadata = self.df.iloc[idx]

        return {'high_res': high_res, 'low_res': low_res, 'rotated': rotated, 'isup': metadata['isup_grade']}

    def __len__(self):
        return len(self.imglist)


class DeepAttentionPL(pl.LightningModule):

    def __init__(self, img_path, df_train, train_idx, val_idx, hparams):
        super().__init__()
        self.train_idx = train_idx
        self.val_idx = val_idx
        c_out = 6

        attention_model = AttentionModelColonCancer(squeeze_channels=True, softmax_smoothing=0)
        feature_model = FeatureModelColonCancer(in_channels=3, out_channels=500)
        classification_head = ClassificationHead(in_channels=500, num_classes=c_out)
        self.ats_model = ATSModel(attention_model, feature_model, classification_head, n_patches=10,
                                  patch_size=80)

        self.hparams = hparams
        self.opt = None
        self.trainset = None
        self.valset = None

        self.img_path = img_path
        self.df_train = df_train

    def forward(self, batch):
        y, attention_map, patches, x_low = self.ats_model(x_low=batch['low_res'], x_high=batch['high_res'])
        return y, attention_map, patches

    def prepare_data(self):
        transform_train = transforms.Compose([transforms.ToPILImage(),
                                              transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                                              transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])

        self.trainset = TiffDataset(self.img_path, self.df_train.iloc[self.train_idx],
                                    transform=transform_train)
        self.valset = TiffDataset(self.img_path, self.df_train.iloc[self.val_idx],
                                  transform=transform_test)

    def train_dataloader(self):
        train_dl = tdata.DataLoader(self.trainset, batch_size=self.hparams.batch_size, shuffle=True,
                                    num_workers=self.hparams.num_workers)
        return train_dl

    def val_dataloader(self):
        val_dl = tdata.DataLoader(self.valset, batch_size=self.hparams.batch_size, shuffle=False,
                                  num_workers=self.hparams.num_workers)
        return [val_dl]

    def cross_entropy_loss(self, logits, gt, attention_map):
        criterion = nn.CrossEntropyLoss()
        entropy_loss_func = MultinomialEntropy(0.01)
        loss = criterion(logits, gt) - entropy_loss_func(attention_map)
        return loss

    def configure_optimizers(self):

        if self.hparams.opt_algo == 'over9000':
            optimizer = Over9000(self.ats_model.parameters(), lr=0.001, weight_decay=3e-6)
            scheduler = FlatCosineAnnealingLR(optimizer, max_iter=self.hparams.epochs,
                                              step_size=self.hparams.step_size)
            return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        logits, attention_map, _ = self(batch)
        loss = self.cross_entropy_loss(logits, batch['isup'], attention_map).unsqueeze(0)
        return {'loss': loss, 'log': {'train_loss': loss}}

    # def training_epoch_end(self, outputs):
    #     preds = torch.cat([out['preds'] for out in outputs], dim=0)
    #     gt = torch.cat([out['gt'] for out in outputs], dim=0)
    #     preds = preds.detach().cpu().numpy()
    #     gt = gt.detach().cpu().numpy()
    #
    #     if self.hparams.task == 'regression' and self.hparams.opt_fit == 'train':
    #         self.opt = OptimizedRounder_v2(6)
    #         self.opt.fit(preds, gt)
    #
    #     return {}

    def validation_step(self, batch, batch_idx):
        logits, attention_map, _ = self(batch)
        loss = self.cross_entropy_loss(logits, batch['isup'], attention_map).unsqueeze(0)
        preds = logits.argmax(1)
        return {'val_loss': loss, 'preds': preds, 'gt': batch['isup']}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.cat([out['val_loss'] for out in outputs], dim=0).mean()
        preds = torch.cat([out['preds'] for out in outputs], dim=0)
        gt = torch.cat([out['gt'] for out in outputs], dim=0)
        preds = preds.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()

        # if self.hparams.task == 'regression':
        #     if self.hparams.use_opt:
        #         if self.opt is None or self.hparams.opt_fit == 'val':
        #             self.opt = OptimizedRounder_v2(6)
        #             self.opt.fit(preds, gt)
        #         preds = self.opt.predict(preds)
        #     else:
        #         preds = np.round(preds)

        kappa = cohen_kappa_score(preds, gt, weights='quadratic')
        print(f'Epoch {self.current_epoch}: {avg_loss:.2f}, kappa: {kappa:.4f}')
        kappa = torch.tensor(kappa)
        tensorboard_logs = {'val_loss': avg_loss, 'kappa': kappa}

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

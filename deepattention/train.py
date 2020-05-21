from utils import dict_to_args
import argparse
import random
import torch
import os
import numpy as np
import pandas as pd
import datetime
from deepattention.deep_attention import DeepAttentionPL
from sklearn.model_selection import StratifiedKFold
import pickle
import pytorch_lightning as pl


parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", default='G:/Datasets/panda/', required=False)
args = parser.parse_args()
ROOT_PATH = args.root_dir

if __name__ == '__main__':

    EPOCHS = 30
    SEED = 33
    BATCH_SIZE = 16
    PRECISION = 32
    NUM_WORKERS = 6

    hparams = {'epochs': EPOCHS,
               'num_workers': NUM_WORKERS,
               'task': 'classification',
               'batch_size': BATCH_SIZE,
               'opt_algo': 'over9000'}

    TRAIN_PATH = ROOT_PATH + f'/train_images'
    CSV_PATH = ROOT_PATH + '/train.csv'

    NAME = 'deepattention'
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

    date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f'Fold {fold + 1}')
        tb_logger = pl.loggers.TensorBoardLogger(save_dir=OUTPUT_DIR,
                                                 name=f'{NAME}' + '-' + date,
                                                 version=f'fold_{fold + 1}')

        checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=tb_logger.log_dir + "/{epoch:02d}-{kappa:.4f}",
                                                           monitor='kappa', mode='max')

        model = DeepAttentionPL(TRAIN_PATH, df_train, train_idx, val_idx, dict_to_args(hparams))
        trainer = pl.Trainer(gpus=[0], max_nb_epochs=EPOCHS, auto_lr_find=False,
                             gradient_clip_val=1,
                             logger=tb_logger,
                             accumulate_grad_batches=1,              # BatchNorm ?
                             checkpoint_callback=checkpoint_callback,
                             precision=PRECISION
                             )
        # lr_finder = trainer.lr_find(model)
        # fig = lr_finder.plot(suggest=True)
        # fig.savefig('lr_plot.png')
        trainer.fit(model)

        # if hparams['use_opt']:
        #     print(model.opt.coefficients())
        #     with open(f'{OUTPUT_DIR}/{NAME}-{date}/fold{fold + 1}_coef.pkl', 'wb') as file:
        #         pickle.dump(file=file, obj=list(np.sort(model.opt.coefficients())))

        # Fold predictions
        torch_model = model.model.eval().to('cuda')
        preds = []
        with torch.no_grad():
            for batch in model.val_dataloader()[0]:
                image = batch['image'].to('cuda')
                pred = torch_model(image)
                preds.append(pred)
        preds = torch.cat(preds, dim=0).squeeze(1).detach().cpu().numpy()
        pd.DataFrame({'val_idx': val_idx, 'preds': preds}).to_csv(
            f'{OUTPUT_DIR}/{NAME}-{date}/fold{fold + 1}_preds.csv', index=False)

        # Todo: One fold training
        # break

# Tests to do:
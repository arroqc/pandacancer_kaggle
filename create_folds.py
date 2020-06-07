import pandas as pd
from sklearn.model_selection import StratifiedKFold
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", default='H:/', required=False)
args = parser.parse_args()
ROOT_PATH = args.root_dir
CSV_PATH = ROOT_PATH + '/train.csv'
SEED = 2020
df_train = pd.read_csv(CSV_PATH)
kfold = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
splits = kfold.split(df_train, df_train['isup_grade'].astype(str) + df_train['data_provider'])

df_train['fold'] = 0
for i, (train_idx, val_idx) in enumerate(splits):
    df_train.loc[val_idx, 'fold'] = i

df_train.to_csv('train.csv', index=False)

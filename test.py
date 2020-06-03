import pandas as pd
import numpy as np

df_train = pd.read_csv('train.csv')

fold_n = df_train['fold'].max()
splits = []
for i in range(0, fold_n + 1):
    train_idx = np.where(df_train['fold'] != i)[0]
    val_idx = np.where(df_train['fold'] == i)[0]
    splits.append((train_idx, val_idx))
    print(val_idx)
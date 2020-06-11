import torch
import torch.nn as nn
from archive.modules import Model
from pathlib import Path
import pickle


#base = "C:/Users/Necka/PycharmProjects\panda/pandacancer_kaggle/lightning_logs/candidates/"
base = "./lightning_logs/"
model_name = "resnext50-20200525-215530"
n_folds = 2
paths = [Path(base + model_name + '/fold_' + str(i)) for i in range(1, n_folds + 1)]
paths = [list(path.glob('*.ckpt'))[0] for path in paths]
print(paths)


class ModelWrapper(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model = Model(c_out=1, **config)

    def forward(self, x):
        return self.model(x)


for i, path in enumerate(paths):

    ckpt = torch.load(path)
    hparams = ckpt['hparams']
    model = ModelWrapper(hparams)
    model.load_state_dict(ckpt['state_dict'])
    torch.save(model.model.state_dict(), base+model_name+'/'+f'fold_{i}.pth')

with open(base+model_name+'/'+f'hparams', 'wb') as file:
    pickle.dump(hparams, file)

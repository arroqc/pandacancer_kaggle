import torch.nn as nn
import torch
from contribs.mish_activation import Mish


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

    def __init__(self, c_out=6, n_tiles=12, tile_size=128, pretrained=True, **kwargs):
        super().__init__()
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
        # m = models.resnet50(pretrained=pretrained)
        c_feature = list(m.children())[-1].in_features
        self.feature_extractor = nn.Sequential(*list(m.children())[:-2])
        self.n_tiles = n_tiles
        self.tile_size = tile_size
        self.head = nn.Sequential(AdaptiveConcatPool2d(),
                                  Flatten(),
                                  nn.Dropout(0.5),
                                  nn.Linear(c_feature * 2, 512),
                                  Mish(),
                                  nn.BatchNorm1d(512),
                                  nn.Dropout(0.5),
                                  nn.Linear(512, c_out))

    def forward(self, x):
        h = x.view(-1, 3, self.tile_size, self.tile_size)
        h = self.feature_extractor(h)
        bn, c, height, width = h.shape
        h = h.view(-1, self.n_tiles, c, height, width).permute(0, 2, 1, 3, 4)\
            .contiguous().view(-1, c, height * self.n_tiles, width)
        h = self.head(h)
        return h

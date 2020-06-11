import torch.nn as nn


class EfficientModel(nn.Module):

    def __init__(self, c_out=5, n_tiles=36, tile_size=224, name='efficientnet-b0'):
        super().__init__()

        from efficientnet_pytorch import EfficientNet
        m = EfficientNet.from_pretrained(name, advprop=True, num_classes=c_out, in_channels=3)
        c_feature = m._fc.in_features
        m._fc = nn.Identity()
        self.feature_extractor = m
        self.n_tiles = n_tiles
        self.tile_size = tile_size
        self.head = nn.Linear(c_feature, c_out)

    def forward(self, x):
        h = self.feature_extractor(x)
        h = self.head(h)
        return h

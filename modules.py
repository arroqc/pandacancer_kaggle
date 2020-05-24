import torch.nn as nn
import torch
from contribs.mish_activation import Mish
from torchvision import models


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

    def __init__(self, c_out=6, n_tiles=12, tile_size=128, backbone='resnext50_semi', head='basic', **kwargs):
        super().__init__()
        if backbone == 'resnext50_semi':
            m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
            remove_range = 2
        elif backbone == 'resnet50':
            m = models.resnet50(pretrained=True)
            remove_range = 2
        elif backbone == 'densenet121':
            m = models.densenet121(pretrained=True)
            remove_range = 1

        c_feature = list(m.children())[-1].in_features
        self.feature_extractor = nn.Sequential(*list(m.children())[:-remove_range])
        self.n_tiles = n_tiles
        self.tile_size = tile_size
        if head == 'basic':
            self.head = BasicHead(c_feature, c_out, n_tiles)
        elif head == 'attention':
            self.head = SelfAttendedHead(c_feature, c_out, n_tiles)
        elif head == 'attention_pool':
            self.head = AttentionPoolHead(c_feature, c_out, n_tiles)

    def forward(self, x):
        h = x.view(-1, 3, self.tile_size, self.tile_size)
        h = self.feature_extractor(h)
        h = self.head(h)

        return h


class TileSelfAttention(nn.Module):

    def __init__(self, c_in, n_tiles, c_out=512):
        super().__init__()
        self.lin_key = nn.Linear(c_in, c_out)
        self.lin_val = nn.Linear(c_in, c_out)
        self.lin_query = nn.Linear(c_in, c_out)
        self.n_tiles = n_tiles
        self.c_out = c_out

    def forward(self, x):
        # Expecting X to be after a per tile pooling so that each tile has some representation of size C
        bn, c = x.shape
        b = bn // self.n_tiles
        keys = self.lin_key(x).reshape(b, self.n_tiles, self.c_out)
        queries = self.lin_query(x).reshape(b, self.n_tiles, self.c_out)
        values = self.lin_val(x).reshape(b, self.n_tiles, self.c_out)

        queries = queries.transpose(1, 2)  # B, C, N
        attention_weights = torch.matmul(keys, queries)  # B,Nk,C * B,C,Nq = B, Nk, Nq
        attention_weights = torch.softmax(attention_weights/(self.c_out ** 0.5), dim=0)  # Key dimension is 0
        out = torch.matmul(values.transpose(1, 2), attention_weights)  # B, C, N * B, N, N = B, C, N

        return out


class SelfAttendedHead(nn.Module):

    def __init__(self, c_in, c_out, n_tiles):
        super().__init__()
        self.attention = TileSelfAttention(c_in * 2, n_tiles, 512)
        self.pool = AdaptiveConcatPool2d()
        self.n_tiles = n_tiles
        self.fc = nn.Sequential(AdaptiveConcatPool2d(),
                                Flatten(),
                                nn.Dropout(0.5),
                                nn.Linear(512 * 2, 512),
                                Mish(),
                                nn.BatchNorm1d(512),
                                nn.Dropout(0.5),
                                nn.Linear(512, c_out))

    def forward(self, x):
        h = self.pool(x).squeeze(2).squeeze(2)  # B * N, C
        h = self.attention(h)  # B, C, N
        h = self.fc(h.unsqueeze(3))  # pool will make B, C * 2, 1, 1

        return h


class BasicHead(nn.Module):

    def __init__(self, c_in, c_out, n_tiles):
        self.n_tiles = n_tiles
        super().__init__()
        self.fc = nn.Sequential(AdaptiveConcatPool2d(),
                                Flatten(),
                                nn.Dropout(0.5),
                                nn.Linear(c_in * 2, 512),
                                Mish(),
                                nn.BatchNorm1d(512),
                                nn.Dropout(0.5),
                                nn.Linear(512, c_out))

    def forward(self, x):

        bn, c, height, width = x.shape
        h = x.view(-1, self.n_tiles, c, height, width).permute(0, 2, 1, 3, 4) \
            .contiguous().view(-1, c, height * self.n_tiles, width)
        h = self.fc(h)
        return h


class AttentionPool(nn.Module):

    def __init__(self, c_in, d):
        super().__init__()
        self.lin_V = nn.Linear(c_in, d)
        self.lin_U = nn.Linear(c_in, d)
        self.lin_w = nn.Linear(d, 1)

    def forward(self, x):
        b, n, c = x.shape
        key = self.lin_V(x)  # b, n, d
        gate = self.lin_U(x)  # b, n, d
        weights = self.lin_w(torch.tanh(key) * torch.sigmoid(gate))  # b, n, 1
        pooled = torch.matmul(x.transpose(1, 2), weights).squeeze(2)   # b, c, n x b, n, 1 => b, c, 1
        return pooled


# class AttentionPoolHead(nn.Module):
#
#     def __init__(self, c_in, c_out, n_tiles, n_head=1):
#         super().__init__()
#         self.attentionheads = nn.ModuleList([AttentionPool(c_in * 2, c_in // (2 * n_head)) for i in range(n_head)])
#         self.maxpool = AdaptiveConcatPool2d()
#         self.n_tiles = n_tiles
#         self.fc = nn.Sequential(nn.Dropout(0.5),
#                                 nn.Linear(c_in * 2, 512),
#                                 Mish(),
#                                 nn.BatchNorm1d(512),
#                                 nn.Dropout(0.5),
#                                 nn.Linear(512, c_out))
#
#     def forward(self, x):
#         bn, c, height, w = x.shape
#         h = self.maxpool(x).squeeze(2).squeeze(2)  # bn, 2c
#         h = h.reshape(-1, self.n_tiles, 2 * c)  # b, n, 2c
#
#         pooled = []
#         for attention in self.attentionheads:
#             pooled.append(attention(h))
#         pooled = torch.stack(pooled, dim=2)
#         pooled = torch.mean(pooled, dim=2)
#
#         return self.fc(pooled)


class AttentionPoolHead(nn.Module):

    def __init__(self, c_in, c_out, n_tiles):
        super().__init__()
        self.maxpool = AdaptiveConcatPool2d()
        self.lin_key = nn.Linear(c_in * 2, c_in // 2)
        self.lin_w = nn.Linear(c_in // 2, 1)
        self.n_tiles = n_tiles
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(c_in * 2, 512),
                                Mish(),
                                nn.BatchNorm1d(512),
                                nn.Dropout(0.5),
                                nn.Linear(512, c_out))

    def compute_attention(self, x):
        keys = self.lin_key(x)
        weights = self.lin_w(torch.tanh(keys))
        weights = weights.reshape(-1, self.n_tiles)
        weights = torch.softmax(weights, dim=1).unsqueeze(2)  # b, n, 1
        return weights

    def forward(self, x):
        bn, c, h, w = x.shape
        h = self.maxpool(x).squeeze(2).squeeze(2)  # bn, c
        weights = self.compute_attention(h)
        h = h.reshape(-1, self.n_tiles, c * 2)
        h = h.transpose(1, 2)
        pooled = torch.matmul(h, weights).squeeze(2)
        return self.fc(pooled)

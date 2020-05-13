import torch.nn as nn
import torch
import math


def init_weights(net, mode='relu', a=0):
    """the weights of conv layer and fully connected layers
    are both initilized with Kaiming Xavier algorithm, In particular,
    we set the parameters to random values uniformly drawn from [-a, a]
    where a = sqrt(6 * (din + dout)), for batch normalization
    layers, y=1, b=0, all bias initialized to 0.
    Simply call init_weights(model) after instantiating it.
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                if mode == 'relu':
                    nn.init.kaiming_uniform_(m.weight, 0)
                elif mode == 'leaky_relu':
                    nn.init.kaiming_uniform_(m.weight, a, mode=mode)
                else:
                    nn.init.kaiming_uniform_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    return net


def split_weights(net):
    """split network weights into two categories,
    one are weights in conv layer and linear layer,
    others are other learnable parameters(conv bias,
    bn weights, bn bias, linear bias)
    Args:
        net: network architecture

    Returns:
        a dictionary of params split into two categories
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)

        else:
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)

    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]


class FlatCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, max_iter, step_size=0.7, last_epoch=-1):
        self.flat_range = int(max_iter * step_size)
        self.T_max = max_iter - self.flat_range
        self.eta_min = 0
        super(FlatCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.flat_range:
            return [base_lr for base_lr in self.base_lrs]
        else:
            cr_epoch = self.last_epoch - self.flat_range
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * (cr_epoch / self.T_max)))
                / 2
                for base_lr in self.base_lrs
            ]

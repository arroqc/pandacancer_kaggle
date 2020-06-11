import torch
from contribs.warmup import GradualWarmupScheduler

model = torch.nn.Conv2d(3, 32, 3)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4 / 10)
# scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30 - 1)
# scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=1,
#                                    after_scheduler=scheduler_cosine)
#
init_lr = 3e-4
warmup_factor = 10
warmup_epo = 1
n_epochs = 30
# scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs-warmup_epo)
# scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_factor, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=init_lr, div_factor=warmup_factor,
                                                total_steps=n_epochs * 1000, pct_start=warmup_epo/n_epochs)

for epoch in range(0, 30):
    print(f'lr: {optimizer.param_groups[0]["lr"]:.7f}')
    for b in range(1000):
        scheduler.step()

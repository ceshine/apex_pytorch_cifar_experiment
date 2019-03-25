import os
import logging

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from helperbot import (
    TriangularLR, WeightDecayOptimizerWrapper, freeze_layers,
    GradualWarmupScheduler
)

from adabound import AdaBound
from baseline import (
    CifarBot, get_cifar10_dataset,
    get_densenet, get_se_resnext, get_resnet,
    get_gpu_memory_map, BOT_TOKEN, CHAT_ID
)
from telegram_sender import telegram_sender


@telegram_sender(token=BOT_TOKEN, chat_id=CHAT_ID)
def train():
    train_dl, valid_dl = get_cifar10_dataset(256)
    steps_per_epoch = len(train_dl)

    model = get_resnet("imagenet")

    optimizer = WeightDecayOptimizerWrapper(optim.Adam(
        model.parameters(), lr=2e-3), 0.1)

    freeze_layers([model], [True])
    freeze_layers([model.last_linear], [False])

    bot = CifarBot(
        model, train_dl, valid_dl,
        optimizer=optimizer, echo=True, avg_window=steps_per_epoch // 5,
        device="cuda:0", clip_grad=1.
    )

    n_epochs = 3
    n_steps = n_epochs * steps_per_epoch
    bot.train(
        n_steps,
        snapshot_interval=steps_per_epoch,
        log_interval=steps_per_epoch // 5,
        keep_n_snapshots=1,
        scheduler=TriangularLR(
            optimizer, 1000, ratio=3, steps_per_cycle=n_steps)
    )
    print(f"GPU Memory Used: {get_gpu_memory_map()} MB")
    bot.remove_checkpoints(keep=1)
    bot.load_model(bot.best_performers[0][1])
    torch.save(bot.model.state_dict(), "cache/stage1.pth")
    bot.remove_checkpoints(keep=0)

    bot.load_model("cache/stage1.pth")
    freeze_layers([model], [False])
    optimizer = WeightDecayOptimizerWrapper(optim.Adam(
        model.parameters(), lr=2e-4), 0.1)
    bot.optimizer = optimizer
    bot.step = 0
    n_epochs = 15
    n_steps = n_epochs * steps_per_epoch
    bot.count_model_parameters()
    bot.train(
        n_steps,
        snapshot_interval=steps_per_epoch,
        log_interval=steps_per_epoch // 5,
        keep_n_snapshots=1,
        scheduler=GradualWarmupScheduler(
            optimizer, 1000, steps_per_epoch * 3,
            after_scheduler=CosineAnnealingLR(
                optimizer, n_steps - (steps_per_epoch * 3)
            )
        )
    )
    print(f"GPU Memory Used: {get_gpu_memory_map()} MB")
    bot.remove_checkpoints(keep=1)
    bot.load_model(bot.best_performers[0][1])
    torch.save(bot.model.state_dict(), "cache/stage2.pth")
    bot.remove_checkpoints(keep=0)


if __name__ == "__main__":
    train()

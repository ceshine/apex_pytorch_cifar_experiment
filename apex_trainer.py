import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
from helperbot import (
    TriangularLR, BaseBot, WeightDecayOptimizerWrapper,
    LearningRateSchedulerCallback
)
from helperbot.metrics import SoftmaxAccuracy
from apex import amp

from adabound import AdaBound
from baseline import (
    CifarBot, get_cifar10_dataset,
    get_densenet, get_se_resnext, get_resnet, get_wide_resnet
    get_gpu_memory_map, BOT_TOKEN, CHAT_ID
)
from telegram_sender import telegram_sender


@telegram_sender(token=BOT_TOKEN, chat_id=CHAT_ID)
def train():
    train_dl, valid_dl = get_cifar10_dataset(batch_size=1024)
    steps_per_epoch = len(train_dl)

    model = get_wide_resnet()

    # optimizer = WeightDecayOptimizerWrapper(optim.SGD(
    #     model.parameters(), lr=0.1,
    #     momentum=0.9, weight_decay=0), 0.05)
    # optimizer = WeightDecayOptimizerWrapper(AdaBound(
    #     model.parameters(), lr=1e-3, final_lr=0.1, gamma=1/steps_per_epoch/2.5, weight_decay=0
    # ), 0.05)
    optimizer = WeightDecayOptimizerWrapper(optim.Adam(
        model.parameters(), lr=1.5e-3), 0.1)
    model, optimizer = amp.initialize(
        model, optimizer, opt_level="O2", keep_batchnorm_fp32=True,
        loss_scale="dynamic"
    )

    n_epochs = 50
    n_steps = n_epochs * steps_per_epoch
    bot = CifarBot(
        model=model, train_loader=train_dl, val_loader=valid_dl,
        optimizer=optimizer, echo=True,
        avg_window=steps_per_epoch // 5,
        criterion=nn.CrossEntropyLoss(),
        device=DEVICE, clip_grad=1.,
        callbacks=[
            LearningRateSchedulerCallback(
                TriangularLR(
                    optimizer, 100, ratio=4, steps_per_cycle=n_steps
                )
            )
        ],
        metrics=[SoftmaxAccuracy()],
        pbar=True,
        use_amp=True
    )
    bot.train(
        n_steps,
        snapshot_interval=steps_per_epoch,
        log_interval=steps_per_epoch // 5,
        keep_n_snapshots=1
    )
    print(f"GPU Memory Used: {get_gpu_memory_map()} MB")
    bot.load_model(bot.best_performers[0][1])
    torch.save(bot.model.state_dict(), "cache/baseline.pth")
    bot.remove_checkpoints(keep=0)


if __name__ == "__main__":
    train()

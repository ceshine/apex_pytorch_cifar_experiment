import os
import types
import logging
import subprocess
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
import pretrainedmodels
from tqdm import tqdm
from helperbot import (
    TriangularLR, BaseBot, WeightDecayOptimizerWrapper, GradualWarmupScheduler,
    LearningRateSchedulerCallback
)
from helperbot.metrics import SoftmaxAccuracy

from adabound import AdaBound
from telegram_sender import telegram_sender
from wide_resnet import Wide_ResNet
from telegram_tokens import BOT_TOKEN, CHAT_ID

TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
TRANSFORM_TEST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
DEVICE = torch.device("cuda")


def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    return int(result)


@dataclass
class CifarBot(BaseBot):
    loss_format = "%.4f"

    @staticmethod
    def extract_prediction(output):
        """Dummy function for compatibility reason"""
        return output


def get_cifar10_dataset(batch_size=512):
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=TRANSFORM_TRAIN)
    train_dl = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    validset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=TRANSFORM_TEST)
    valid_dl = torch.utils.data.DataLoader(
        validset, batch_size=batch_size*2, shuffle=False, num_workers=1, pin_memory=True)
    return train_dl, valid_dl


def get_resnet(pretrained=None):
    model = pretrainedmodels.__dict__["resnet34"](
        pretrained=pretrained)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.last_linear = nn.Linear(model.last_linear.in_features, 10)
    model.to(DEVICE)
    return model


def get_densenet(pretrained=None):
    model = pretrainedmodels.__dict__["densenet121"](
        pretrained=pretrained)
    model.last_linear = nn.Linear(model.last_linear.in_features, 10)

    def logits(self, features):
        x = F.relu(features, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x
    # Modify methods
    model.logits = types.MethodType(logits, model)
    model.to(DEVICE)
    return model


def get_se_resnext(pretrained=None):
    model = pretrainedmodels.__dict__["se_resnext50_32x4d"](
        pretrained=pretrained)
    model.last_linear = nn.Linear(model.last_linear.in_features, 10)
    model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    model.to(DEVICE)
    return model


def get_wide_resnet():
    return Wide_ResNet(28, 10, 0.3, 10).to(DEVICE)


@telegram_sender(token=BOT_TOKEN, chat_id=CHAT_ID)
def train():
    train_dl, valid_dl = get_cifar10_dataset(128)
    steps_per_epoch = len(train_dl)

    model = get_wide_resnet()

    n_epochs = 25
    n_steps = n_epochs * steps_per_epoch

    # optimizer = WeightDecayOptimizerWrapper(optim.SGD(
    #     model.parameters(), lr=0.1,
    #     momentum=0.9, weight_decay=0), 0.02)
    # optimizer = WeightDecayOptimizerWrapper(AdaBound(
    #     model.parameters(), lr=1e-3, final_lr=0.1, gamma=1/300, weight_decay=0
    # ), 0.001)
    optimizer = WeightDecayOptimizerWrapper(optim.Adam(
        model.parameters(), lr=1.5e-3), 0.1)
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
        monitor_metric="accuracy"
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

import os
import types
import logging
import subprocess

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
    TriangularLR, BaseBot, WeightDecayOptimizerWrapper, GradualWarmupScheduler
)

from adabound import AdaBound
from telegram_sender import telegram_sender

TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
TRANSFORM_TEST = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
BOT_TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID = int(os.environ["CHAT_ID"])


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


class CifarBot(BaseBot):
    def __init__(self, model, train_dl, valid_dl, *, optimizer, clip_grad=0,
                 avg_window=200, log_dir="./cache/logs/", log_level=logging.INFO,
                 checkpoint_dir="./cache/model_cache/", echo=False,
                 device="cuda:0", use_tensorboard=False, use_amp=False):
        super().__init__(
            model, train_dl, valid_dl,
            optimizer=optimizer,
            clip_grad=clip_grad,
            avg_window=avg_window,
            log_dir=log_dir,
            log_level=log_level,
            checkpoint_dir=checkpoint_dir,
            batch_idx=0,
            echo=echo,
            device=device,
            use_tensorboard=use_tensorboard,
            use_amp=use_amp
        )
        self.criterion = nn.CrossEntropyLoss()
        self.loss_format = "%.4f"

    @staticmethod
    def extract_prediction(output):
        """Dummy function for compatibility reason"""
        return output

    def eval(self, loader):
        self.model.eval()
        preds, ys = [], []
        losses, weights = [], []
        with torch.set_grad_enabled(False):
            for *input_tensors, y in tqdm(loader):
                input_tensors = [x.to(self.device) for x in input_tensors]
                output = self.model(*input_tensors)
                batch_loss = self.criterion(
                    self.extract_prediction(output),
                    y.to(self.device)
                )
                preds.append(self.extract_prediction(output).cpu())
                ys.append(y.cpu().numpy())
                losses.append(batch_loss.data.cpu().numpy())
                weights.append(y.size(self.batch_idx))
        loss = np.average(losses, weights=weights)
        final_preds = np.argmax(torch.cat(preds).numpy(), axis=1)
        final_ys = np.concatenate(ys)
        accuracy = np.sum(final_ys == final_preds) / len(final_preds)
        self.logger.info("Accuracy %.4f", accuracy)
        self.logger.info("Log Loss %.4f", loss)
        self.logger.tb_scalars(
            "losses", {"val": loss},  self.step)
        return accuracy * -1


def get_cifar10_dataset(batch_size=512):
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=TRANSFORM_TRAIN)
    train_dl = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    validset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=TRANSFORM_TEST)
    valid_dl = torch.utils.data.DataLoader(
        validset, batch_size=batch_size*2, shuffle=False, num_workers=1, pin_memory=True)
    return train_dl, valid_dl


def get_resnet(pretrained=None):
    model = pretrainedmodels.__dict__["resnet152"](
        pretrained=pretrained)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.last_linear = nn.Linear(model.last_linear.in_features, 10)
    model.to("cuda:0")
    return model


def get_densenet(pretrained=None):
    model = pretrainedmodels.__dict__["densenet161"](
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
    model.to("cuda:0")
    return model


def get_se_resnext(pretrained=None):
    model = pretrainedmodels.__dict__["se_resnext50_32x4d"](
        pretrained=pretrained)
    model.last_linear = nn.Linear(model.last_linear.in_features, 10)
    model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    model.to("cuda:0")
    return model


@telegram_sender(token=BOT_TOKEN, chat_id=CHAT_ID)
def train():
    train_dl, valid_dl = get_cifar10_dataset(512)
    steps_per_epoch = len(train_dl)

    model = get_densenet()

    # optimizer = WeightDecayOptimizerWrapper(optim.SGD(
    #     model.parameters(), lr=0.1,
    #     momentum=0.9, weight_decay=0), 0.02)
    # optimizer = WeightDecayOptimizerWrapper(AdaBound(
    #     model.parameters(), lr=1e-3, final_lr=0.1, gamma=1/300, weight_decay=0
    # ), 0.001)
    optimizer = WeightDecayOptimizerWrapper(optim.Adam(
        model.parameters(), lr=1.5e-3), 0.1)
    bot = CifarBot(
        model, train_dl, valid_dl,
        optimizer=optimizer, echo=True, avg_window=steps_per_epoch // 5,
        device="cuda:0", clip_grad=1.
    )

    n_epochs = 25
    n_steps = n_epochs * steps_per_epoch
    bot.train(
        n_steps,
        snapshot_interval=steps_per_epoch,
        log_interval=steps_per_epoch // 5,
        keep_n_snapshots=3,
        # scheduler=GradualWarmupScheduler(
        #     optimizer, 1000, steps_per_epoch * 2,
        #     after_scheduler=CosineAnnealingLR(
        #         optimizer, n_steps - (steps_per_epoch * 2)
        #     )
        # )
        scheduler=TriangularLR(
            optimizer, 1000, ratio=3, steps_per_cycle=n_steps)
    )
    print(f"GPU Memory Used: {get_gpu_memory_map()} MB")
    bot.remove_checkpoints(keep=1)
    bot.load_model(bot.best_performers[0][1])
    torch.save(bot.model.state_dict(), "cache/baseline.pth")
    bot.remove_checkpoints(keep=0)


if __name__ == "__main__":
    train()

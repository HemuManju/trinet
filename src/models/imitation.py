import torch
import torch.nn as nn
import pytorch_lightning as pl


import torchmetrics

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WeightedMSE(torch.nn.MSELoss):
    def __init__(self, weights=None):
        super().__init__(reduction='none')
        self.weights = weights

    def forward(self, input, target):
        if self.weights is not None:
            return torch.mean(
                torch.sum(self.weights * super().forward(input, target), dim=1)
            )
        else:
            return torch.mean(super().forward(input, target))


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


class AuxiliaryTraining(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super(AuxiliaryTraining, self).__init__()
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader
        self.accuracy_traffic = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.accuracy_vehicle_dist = torchmetrics.Accuracy(
            task="multiclass", num_classes=5
        )

        # Save hyperparameters
        # self.save_hyperparameters(self.h_params)

    def forward(self, x, command):
        output = self.net.forward(x, command)
        return output

    def training_step(self, batch, batch_idx):
        images, command, action = (
            batch[0],
            batch[1],
            batch[2],
        )

        # Predict and calculate loss
        output = self.forward(images, command)
        criterion = nn.MSELoss()

        # Classification
        criterion_traffic = nn.CrossEntropyLoss()
        criterion_vehicle_dist = nn.CrossEntropyLoss()

        loss = (
            criterion(output[0][0], action[0])
            + criterion_traffic(output[1], action[2])
            + criterion_vehicle_dist(output[2], action[3])
        )
        self.accuracy_traffic(output[1].argmax(dim=1), action[2])
        self.accuracy_vehicle_dist(output[2].argmax(dim=1), action[3])

        self.log(
            'train_traffic_accuracy',
            self.accuracy_traffic,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            'train_vehicle_dist_accuracy',
            self.accuracy_vehicle_dist,
            on_step=False,
            on_epoch=True,
        )
        self.log('losses/train_loss', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, command, action = (
            batch[0],
            batch[1],
            batch[2],
        )

        # Predict and calculate loss
        output = self.forward(images, command)
        criterion = nn.MSELoss()

        # Classification
        criterion_traffic = nn.CrossEntropyLoss()
        criterion_vehicle_dist = nn.CrossEntropyLoss()

        loss = (
            criterion(output[0][0], action[0])
            + criterion_traffic(output[1], action[2])
            + criterion_vehicle_dist(output[2], action[3])
        )

        self.accuracy_traffic(output[1].argmax(dim=1), action[2])
        self.accuracy_vehicle_dist(output[2].argmax(dim=1), action[3])

        self.log(
            'val_traffic_accuracy', self.accuracy_traffic, on_step=False, on_epoch=True,
        )
        self.log(
            'val_vehicle_dist_accuracy',
            self.accuracy_vehicle_dist,
            on_step=False,
            on_epoch=True,
        )
        self.log('losses/val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def train_dataloader(self):
        return self.data_loader['training']

    def val_dataloader(self):
        return self.data_loader['validation']

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.h_params['LEARNING_RATE'])
        lr_scheduler = ReduceLROnPlateau(
            optimizer, patience=5, factor=0.95, verbose=True
        )

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'losses/val_loss',
        }

        if self.h_params['use_scheduler']:
            return [optimizer], [scheduler]
        else:
            return [optimizer]


class Imitation(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super(Imitation, self).__init__()
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader

        # Save hyperparameters
        # self.save_hyperparameters(self.h_params)

    def forward(self, x, command, kalman):
        output = self.net.forward(x, command, kalman)
        return output

    def training_step(self, batch, batch_idx):
        images, command, action, kalman = batch[0], batch[1], batch[2], batch[3]

        # Predict and calculate loss
        output = self.forward(images, command, kalman)
        # criterion1 = RMSELoss()
        criterion = nn.MSELoss()
        # loss1 = criterion1(output[0], action[0]) + criterion2(output[0], action[0])
        # loss2 = criterion1(output[1], action[1]) + criterion2(output[1], action[1])

        # loss = loss1 + loss2
        # loss = criterion2(output[0], action[0])
        loss = criterion(output[0], action[0])

        self.log('losses/train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, command, action, kalman = batch[0], batch[1], batch[2], batch[3]

        # Predict and calculate loss
        output = self.forward(images, command, kalman)
        # criterion1 = RMSELoss()
        criterion = nn.MSELoss()
        # loss1 = criterion1(output[0], action[0]) + criterion2(output[0], action[0])
        # loss2 = criterion1(output[1], action[1]) + criterion2(output[1], action[1])

        # loss = loss1 + loss2
        # loss = criterion2(output[0], action[0])
        loss = criterion(output[0], action[0])

        self.log('losses/val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def train_dataloader(self):
        return self.data_loader['training']

    def val_dataloader(self):
        return self.data_loader['validation']

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.h_params['LEARNING_RATE'])
        lr_scheduler = ReduceLROnPlateau(
            optimizer, patience=5, factor=0.95, verbose=True
        )

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'losses/val_loss',
        }

        if self.h_params['use_scheduler']:
            return [optimizer], [scheduler]
        else:
            return [optimizer]

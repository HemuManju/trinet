import torch
import torch.nn as nn
import pytorch_lightning as pl

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
        criterion1 = RMSELoss()
        criterion2 = nn.MSELoss()
        loss1 = criterion1(output[0], action[0]) + criterion2(output[0], action[0])
        loss2 = criterion1(output[1], action[1]) + criterion2(output[1], action[1])

        loss = loss1 + loss2
        # loss = criterion2(output[0], action[0])

        self.log('losses/train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, command, action, kalman = batch[0], batch[1], batch[2], batch[3]

        # Predict and calculate loss
        output = self.forward(images, command, kalman)
        criterion1 = RMSELoss()
        criterion2 = nn.MSELoss()
        loss1 = criterion1(output[0], action[0]) + criterion2(output[0], action[0])
        loss2 = criterion1(output[1], action[1]) + criterion2(output[1], action[1])

        loss = loss1 + loss2
        # loss = criterion2(output[0], action[0])

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

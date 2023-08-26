import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_msssim import MS_SSIM
from piq import MultiScaleSSIMLoss

import torchmetrics

from .utils import ChamferDistance, calc_ssim_kernel_size


class Autoencoder(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super(Autoencoder, self).__init__()
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader

        # Save hyperparameters
        self.save_hyperparameters(self.h_params)

    def forward(self, x):
        output = self.net.forward(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Predict and calculate loss
        output, embedding = self.forward(x)

        k = calc_ssim_kernel_size(self.h_params['image_resize'], levels=5)
        criterion = MultiScaleSSIMLoss(kernel_size=11)
        criterion_l1 = nn.MSELoss()

        loss = criterion(output, y)  # + criterion_l1(output, y)

        self.log('losses/train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Predict and calculate loss
        output, embedding = self.forward(x)

        k = calc_ssim_kernel_size(self.h_params['image_resize'], levels=5)
        criterion = MultiScaleSSIMLoss(kernel_size=11)
        criterion_l1 = nn.MSELoss()

        loss = criterion(output, y)  # + criterion_l1(output, y)
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


class SemanticSegmentation(Autoencoder):
    def __init__(self, hparams, net, data_loader):
        super().__init__(hparams, net, data_loader)
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader

        # Save hyperparameters
        self.save_hyperparameters(self.h_params)

    def training_step(self, batch, batch_idx):
        x, labels = batch

        # Predict and calculate loss
        output, embeddings = self.forward(x)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output.squeeze(1), labels)

        self.log('losses/train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch

        # Predict and calculate loss
        output, embeddings = self.forward(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output.squeeze(1), labels)

        self.log('losses/val_loss', loss, on_step=False, on_epoch=True)
        return loss


class SemanticAuxSegmentation(Autoencoder):
    def __init__(self, hparams, net, data_loader):
        super().__init__(hparams, net, data_loader)
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader
        self.accuracy_traffic = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.accuracy_vehicle_dist = torchmetrics.Accuracy(
            task="multiclass", num_classes=5
        )

        # Save hyperparameters
        self.save_hyperparameters(self.h_params)

    def training_step(self, batch, batch_idx):
        x, labels, action = batch[0], batch[1], batch[2]

        # Predict and calculate loss
        output, traffic_out, distance_out, embedding = self.forward(x)

        criterion = nn.CrossEntropyLoss()

        # Classification
        criterion_traffic = nn.CrossEntropyLoss()
        criterion_vehicle_dist = nn.CrossEntropyLoss()

        loss = (
            criterion(output.squeeze(1), labels)
            + criterion_traffic(traffic_out, action[1])
            + criterion_vehicle_dist(distance_out, action[2])
        )

        self.accuracy_traffic(traffic_out.argmax(dim=1), action[1])
        self.accuracy_vehicle_dist(distance_out.argmax(dim=1), action[2])

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
        x, labels, action = batch[0], batch[1], batch[2]

        # Predict and calculate loss
        output, traffic_out, distance_out, embedding = self.forward(x)
        criterion = nn.CrossEntropyLoss()

        # Classification
        criterion_traffic = nn.CrossEntropyLoss()
        criterion_vehicle_dist = nn.CrossEntropyLoss()

        loss = (
            criterion(output.squeeze(1), labels)
            + criterion_traffic(traffic_out, action[1])
            + criterion_vehicle_dist(distance_out, action[2])
        )

        self.accuracy_traffic(traffic_out.argmax(dim=1), action[1])
        self.accuracy_vehicle_dist(distance_out.argmax(dim=1), action[2])

        self.log(
            'val_traffic_accuracy',
            self.accuracy_traffic,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            'val_vehicle_dist_accuracy',
            self.accuracy_vehicle_dist,
            on_step=False,
            on_epoch=True,
        )
        self.log('losses/val_loss', loss, on_step=False, on_epoch=True)
        return loss


class RNNSegmentation(Autoencoder):
    def __init__(self, hparams, net, data_loader):
        super().__init__(hparams, net, data_loader)
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader

        # Save hyperparameters
        self.save_hyperparameters(self.h_params)

    def training_step(self, batch, batch_idx):
        x, labels = batch

        # Predict and calculate loss
        output, embeddings = self.forward(x)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)

        self.log('losses/train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch

        # Predict and calculate loss
        output, embeddings = self.forward(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)  # Target shape: [batch, seq length, H, W]

        self.log('losses/val_loss', loss, on_step=False, on_epoch=True)
        return loss


class RNNEncoder(Autoencoder):
    def __init__(self, hparams, net, data_loader):
        super().__init__(hparams, net, data_loader)
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader

        self.batch_size = self.h_params['BATCH_SIZE']
        self.seq_length = self.h_params['seq_length'] - 1
        self.image_size = self.h_params['image_resize']

        # Save hyperparameters
        self.save_hyperparameters(self.h_params)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        batch_size, timesteps, C, H, W = labels.size()

        # Predict and calculate loss
        output, embeddings = self.forward(x)
        criterion = MultiScaleSSIMLoss(kernel_size=11)
        loss = criterion(
            output.reshape(batch_size * timesteps, C, H, W),
            labels.reshape(batch_size * timesteps, C, H, W),
        )

        self.log('losses/train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        batch_size, timesteps, C, H, W = labels.size()

        # Predict and calculate loss
        output, embeddings = self.forward(x)
        criterion = MultiScaleSSIMLoss(kernel_size=11)
        loss = criterion(
            output.reshape(batch_size * timesteps, C, H, W),
            labels.reshape(batch_size * timesteps, C, H, W),
        )

        self.log('losses/val_loss', loss, on_step=False, on_epoch=True)
        return loss


class KalmanRNNEncoder(RNNEncoder):
    def __init__(self, hparams, net, data_loader):
        super().__init__(hparams, net, data_loader)
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader

        self.batch_size = self.h_params['BATCH_SIZE']
        self.seq_length = self.h_params['seq_length'] - 1
        self.image_size = self.h_params['image_resize']

        # Save hyperparameters
        self.save_hyperparameters(self.h_params)

    def forward(self, x, kalman):
        output = self.net.forward(x, kalman)
        return output

    def training_step(self, batch, batch_idx):
        x, y, kalman = batch
        batch_size, timesteps, C, H, W = y.size()

        # Predict and calculate loss
        output, out_ae, rnn_embeddings = self.forward(x, kalman)
        criterion = MultiScaleSSIMLoss(kernel_size=11)
        loss = criterion(
            output.reshape(batch_size * timesteps, C, H, W),
            y.reshape(batch_size * timesteps, C, H, W),
        )

        self.log('losses/train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, kalman = batch
        batch_size, timesteps, C, H, W = y.size()

        # Predict and calculate loss
        output, out_ae, rnn_embeddings = self.forward(x, kalman)
        criterion = MultiScaleSSIMLoss(kernel_size=11)
        loss = criterion(
            output.reshape(batch_size * timesteps, C, H, W),
            y.reshape(batch_size * timesteps, C, H, W),
        )

        self.log('losses/val_loss', loss, on_step=False, on_epoch=True)
        return loss

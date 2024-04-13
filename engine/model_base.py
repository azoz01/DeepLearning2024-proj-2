from abc import ABC, abstractmethod
from datetime import datetime

import pytorch_lightning as pl
import torch

from .metrics import accuracy, balanced_accuracy


class LightningBaseModule(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()
        self.hparams["cls_name"] = type(self).__name__
        self.hparams["timestamp"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

    def training_step(self, batch):
        X, y = batch
        predictions = self.forward(X)
        loss = self.loss(predictions, y)
        self.log("train_step_loss", loss, prog_bar=True)
        return {"loss": loss, "predictions": predictions}

    def on_train_epoch_start(self, *args, **kwargs):
        self.training_labels = []
        self.training_predictions = []

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.training_predictions.append(outputs["predictions"])
        self.training_labels.append(batch[1])

    def on_train_epoch_end(self, *args, **kwargs):
        training_labels = torch.concat(self.training_labels, dim=0)
        training_predictions = torch.concat(self.training_predictions, dim=0)
        self.log(
            "train_accuracy",
            accuracy(training_labels, training_predictions.argmax(dim=1)),
        )
        self.log(
            "train_balanced_accuracy",
            balanced_accuracy(training_labels, training_predictions.argmax(dim=1)),
        )

    def validation_step(self, batch):
        X, y = batch
        predictions = self.forward(X)
        loss = self.loss(predictions, y)
        self.log(
            "val_step_loss",
            loss,
            prog_bar=True,
        )
        return {"loss": loss, "predictions": predictions}

    def on_validation_epoch_start(self, *args, **kwargs):
        self.validation_labels = []
        self.validation_predictions = []

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.validation_predictions.append(outputs["predictions"])
        self.validation_labels.append(batch[1])

    def on_validation_epoch_end(self, *args, **kwargs):
        validation_labels = torch.concat(self.validation_labels, dim=0)
        validation_predictions = torch.concat(
            self.validation_predictions, dim=0
        )
        self.log(
            "val_accuracy",
            accuracy(validation_labels, validation_predictions.argmax(dim=1)),
        )
        self.log(
            "val_balanced_accuracy",
            balanced_accuracy(validation_labels, validation_predictions.argmax(dim=1)),
        )

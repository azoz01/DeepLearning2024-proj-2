import pytorch_lightning as pl
import torch

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from .metrics import accuracy


class LightningBaseModule(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()
        self.tensorboard_logger = self.__get_tensorboard_logger()

    def __get_tensorboard_logger(self):
        model_name = type(self).__name__
        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        logger_path = Path(f"results/{model_name}/{timestamp}")
        logger_path.mkdir(exist_ok=True, parents=True)
        return SummaryWriter(logger_path)

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
        self.__log_to_all_loggers(
            "train_accuracy",
            accuracy(training_labels, training_predictions.argmax(dim=1)),
        )

    def validation_step(self, batch):
        X, y = batch
        predictions = self.forward(X)
        loss = self.loss(predictions, y)
        self.log("val_step_loss", loss, prog_bar=True)
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
        self.__log_to_all_loggers(
            "val_accuracy",
            accuracy(validation_labels, validation_predictions.argmax(dim=1)),
        )

    def __log_to_all_loggers(self, name, value):
        self.log(name, value)
        self.tensorboard_logger.add_scalar(name, value, self.current_epoch)

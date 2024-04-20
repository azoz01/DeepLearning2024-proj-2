from copy import deepcopy
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from engine.data import get_data_loader
from engine.lstm import LSTMCommandRecognition


def _get_callback():
    callbacks = [
        EarlyStopping(
            monitor="val_balanced_accuracy",
            mode="max",
            patience=10,
            min_delta=1e-5,
        ),
        ModelCheckpoint(
            save_top_k=1,
            monitor="val_accuracy",
            filename="model-{epoch}-{val_accuracy:.2f}",
            mode="max",
            every_n_epochs=1,
        ),
    ]

    return callbacks


torch.set_float32_matmul_precision("medium")
pl.seed_everything(123)

train_loader = get_data_loader("train", undersample_majority=True)
val_loader = get_data_loader("val")
test_loader = get_data_loader("test")

base_hp: Dict[str, any] = {
    "input_size": 80,
    "output_size": 12,
    "hidden_size": 512,
    "lstm_units": 1,
    "bidirectional": False,
    "num_used_state": 1,
    "dropout": 0.3,
    "use_layer_norm": True,
    "learning_rate": 1e-4,
}

modifications: Dict[str, Tuple[str, any]] = {
    "bigger_hidden_size": ("hidden_size", 512 * 4),
    "more_lstm_units": ("lstm_units", 4),
    "bidirectional": ("bidirectional", True),
    "wholle_sequence": ("num_used_state", 80),
    "bigger_dropout": ("dropout", 0.5),
    "without_layer_norm": ("use_layer_norm", False),
}

for modification_name, modification in modifications.items():
    for i in range(5):
        print(i)
        pl.seed_everything(i)
        callbacks = _get_callback()
        hp = deepcopy(base_hp)
        hp[modification[0]] = modification[1]

        model = LSTMCommandRecognition(**hp).cuda()

        trainer = pl.Trainer(
            max_epochs=50,
            callbacks=callbacks,
            gradient_clip_val=1,
            gradient_clip_algorithm="norm",
            default_root_dir=f"results/lstm/{modification_name}",
            deterministic=True,
        )
        trainer.fit(model, train_loader, val_loader)

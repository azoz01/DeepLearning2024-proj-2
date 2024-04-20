from typing import Dict, List

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from engine.model_base import LightningBaseModule


class LSTMCommandRecognition(LightningBaseModule):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        lstm_units: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False,
        num_used_state: int = 1,
        use_layer_norm: bool = True,
        learning_rate: float = 1e-4,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lstm_unit = lstm_units
        self.dropout = dropout

        assert not bidirectional
        self.bidirectional = bidirectional
        self.num_used_state = num_used_state
        self.learning_rate = learning_rate

        self.lstm = nn.Sequential(
            *[
                self._generate_lstm_layer(
                    {
                        "input_size": input_size if i == 0 else hidden_size,
                        "hidden_size": hidden_size,
                        "num_layers": 1,
                        "dropout": dropout,
                        "bidirectional": bidirectional,
                        "batch_first": True,
                    },
                    use_layer_norm=use_layer_norm,
                )
                for i in range(self.lstm_unit)
            ]
        )

        lstm_output_size = (
            hidden_size if not bidirectional else hidden_size * 2
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(lstm_output_size * num_used_state, output_size)
        self.softmax = nn.Softmax()

        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, X):

        out = self.lstm(X)

        used_states = [
            (i + 1) * out.shape[1] // self.num_used_state - 1
            for i in range(self.num_used_state)
        ]
        out = self.flatten(out[:, used_states, :])
        out = self.linear(out)
        out = self.softmax(out)

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, patience=5)
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_accuracy",
                "frequency": 1,
            }
        ]

    def _generate_lstm_layer(
        self, lstm_params: Dict[str, any], use_layer_norm: bool
    ) -> List:
        lstm = LSTM(
            **lstm_params,
        )
        if use_layer_norm:
            layer_norm = nn.LayerNorm(
                self.hidden_size * (2 if self.bidirectional else 1)
            )
            return nn.Sequential(lstm, layer_norm)

        else:
            return nn.Sequential(lstm)


class LSTM(nn.LSTM):

    def forward(self, X):
        out, _ = super().forward(X)
        return out

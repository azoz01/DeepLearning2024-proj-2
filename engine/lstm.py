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
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lstm_unit = lstm_units
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_used_state = num_used_state

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=lstm_units,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
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

        out, _ = self.lstm(X)

        used_states = [
            (i + 1) * out.shape[1] // self.num_used_state - 1
            for i in range(self.num_used_state)
        ]
        out = self.flatten(out[:, used_states, :])
        out = self.linear(out)
        out = self.softmax(out)

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, patience=5)
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_accuracy",
                "frequency": 1,
            }
        ]

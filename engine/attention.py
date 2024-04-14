import numpy as np
import torch
import torch.nn.functional as F

from functools import lru_cache
from torch import nn
from torch.optim.lr_scheduler import LinearLR

from .model_base import LightningBaseModule


class ConvolutionalBlock(nn.Module):
    def __init__(self, n_mels, n_state):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            n_state, n_state, kernel_size=3, stride=2, padding=1
        )

    def forward(self, X):
        X = X.permute(0, 2, 1)
        X = F.gelu(self.conv1(X))
        X = F.gelu(self.conv2(X))
        return X.permute(0, 2, 1)


class AttentionBlock(nn.Module):
    def __init__(self, hidden_state_size: int, n_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_state_size, n_heads, batch_first=True
        )
        self.attention_layer_norm = nn.LayerNorm(hidden_state_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_state_size, 3 * hidden_state_size),
            nn.GELU(),
            nn.Linear(3 * hidden_state_size, hidden_state_size),
        )
        self.mlp_layer_norm = nn.LayerNorm(hidden_state_size)

    def forward(self, X):
        X = self.attention_layer_norm(X)
        x = X + self.attention(X, X, X)[0]
        X = X + self.mlp(self.mlp_layer_norm(X))
        return X


class AudioEncoder(nn.Module):
    def __init__(self, n_mels, n_hidden, n_heads, n_layers):
        super().__init__()
        self.convolutional_block = ConvolutionalBlock(n_mels, n_hidden)
        self.attention = nn.ModuleList(
            [AttentionBlock(n_hidden, n_heads) for _ in range(n_layers)]
        )
        self.layer_norm_post = nn.LayerNorm(n_hidden)

    def forward(self, X):
        X = self.convolutional_block(X)
        X = X + self.__positional_embedding(X.shape[1], X.shape[2]).cuda()
        for block in self.attention:
            X = block(X)
        X = self.layer_norm_post(X)
        return X
    
    @lru_cache
    def __positional_embedding(self, length, channels, max_timescale=10000):
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class AttentionAudioClassifier(LightningBaseModule):
    def __init__(
        self,
        n_classes,
        sequence_length,
        n_mels,
        n_hidden,
        n_heads,
        n_attention_layers,
    ):
        super().__init__()
        self.encoder = AudioEncoder(
            n_mels, n_hidden, n_heads, n_attention_layers
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                sequence_length // 2 * n_hidden,
                256
            ),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(
                256,
                n_classes
            ),
            nn.Softmax(dim=1),
        )
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, X):
        X = self.encoder(X)
        X = self.classifier(X)
        return X

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-1)
        scheduler = LinearLR(optimizer)
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_accuracy",
                "frequency": 1,
            }
        ]

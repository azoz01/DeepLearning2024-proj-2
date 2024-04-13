import json
import pickle as pkl
from typing import Literal

import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader


def _retreive_data(sample: Literal["train", "val", "test"]) -> DataLoader:
    data = torch.load(f"data/converted/{sample}_data.pt")
    with open(f"data/converted/{sample}_labels.json") as f:
        labels = json.load(f)
    labels = torch.LongTensor(labels)
    data = data.cuda()
    labels = labels.cuda()

    return data, labels


def _create_loader(
    data: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    dataset = data_utils.TensorDataset(data, labels)
    loader = data_utils.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return loader


def get_data_loader(sample: Literal["train", "val", "test"]) -> DataLoader:
    """
    Return all 12 classes
    """
    data, labels = _retreive_data(sample)

    shuffle = True if sample == "train" else False
    return _create_loader(data, labels, batch_size=64, shuffle=shuffle)


def get_commands_loader(sample: Literal["train", "val", "test"]) -> DataLoader:
    """
    Return only commands
    """
    data, labels = _retreive_data(sample)

    with open("data/converted/encoder.pkl", "rb") as f:
        encoder = pkl.load(f)

    cls_names = encoder.inverse_transform(list(range(12)))

    commands_idx = [
        idx
        for idx, cls_name in zip(range(12), cls_names)
        if cls_name not in {"unknown", "silence"}
    ]

    commands_rows = torch.isin(labels, torch.Tensor(commands_idx).cuda())

    data, labels = data[commands_rows], labels[commands_rows]

    shuffle = True if sample == "train" else False
    return _create_loader(data, labels, batch_size=64, shuffle=shuffle)


def get_main_classes_loader(
    sample: Literal["train", "val", "test"]
) -> DataLoader:
    """
    Cast to three categories: commands, unknown, silence
    """
    data, labels = _retreive_data(sample)

    with open("data/converted/encoder.pkl", "rb") as f:
        encoder = pkl.load(f)

    cls_names = list(encoder.inverse_transform(list(range(12))))

    commands_idx = [
        idx
        for idx, cls_name in zip(range(12), cls_names)
        if cls_name not in {"unknown", "silence"}
    ]

    unknown_idx = cls_names.index("unknown")
    silence_idx = cls_names.index("silence")

    commands_rows = torch.isin(labels, torch.Tensor(commands_idx).cuda())

    labels[commands_rows] = 0
    labels[labels == unknown_idx] = 1
    labels[labels == silence_idx] = 2

    shuffle = True if sample == "train" else False
    return _create_loader(data, labels, batch_size=64, shuffle=shuffle)

import json
import numpy as np
import pickle as pkl
import torch
import torch.utils.data as data_utils

from torch.utils.data import DataLoader
from typing import Literal

from collections import Counter

def _retreive_data(
    sample: Literal["train", "val", "test"],
    undersample_majority: bool = False
) -> DataLoader:
    data = torch.load(f"data/converted/{sample}_data.pt")
    with open(f"data/converted/{sample}_labels.json") as f:
        labels = json.load(f)
    if undersample_majority:
        data, labels = __undersample_dataset(data, labels)
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


def get_data_loader(
    sample: Literal["train", "val", "test"],
    undersample_majority: bool = False
) -> DataLoader:
    """
    Return all 12 classes
    """
    data, labels = _retreive_data(sample, undersample_majority)

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


def __undersample_dataset(data, labels):
    counter = Counter(labels)
    _, voice_command_class_count = counter.most_common(3)[-1]
    for label, observation_count in counter.most_common(2):
        not_label_examples = data[np.array(labels) != label]
        label_examples = data[np.array(labels) == label]
        label_examples = label_examples[
            np.random.choice(len(label_examples), size=voice_command_class_count)
        ]
        data = torch.concat([not_label_examples, label_examples], dim=0)

        labels = [el for el in labels if el != label]
        labels = labels + [label]*voice_command_class_count
    return data, labels


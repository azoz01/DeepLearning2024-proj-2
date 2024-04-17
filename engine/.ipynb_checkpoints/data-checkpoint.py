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
    commands_idx_mapping = dict(zip(commands_idx, range(len(commands_idx))))

    commands_rows = torch.isin(labels, torch.LongTensor(commands_idx).cuda())

    data, labels = data[commands_rows], labels[commands_rows]
    labels = torch.LongTensor([commands_idx_mapping[label] for label in labels.cpu().numpy()])

    shuffle = sample == "train"
    return _create_loader(data, labels, batch_size=64, shuffle=shuffle)


def get_main_classes_loader(
    sample: Literal["train", "val", "test"], oversample_silence: bool = False
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

    if oversample_silence:
        silence_mask = labels.cpu().numpy() == 2
        silence_loc = np.where(silence_mask)[0]
        silence_loc = np.random.choice(
            silence_loc,
            size=(len(silence_mask) - silence_mask.sum()) // 2 - silence_mask.sum(),
            replace=True
        )
        additional_silence = data[silence_loc]
        additional_labels = [2]*len(silence_loc)
        data = torch.concat([data, additional_silence])
        labels = torch.concat([labels, torch.LongTensor(additional_labels).cuda()])

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


def get_label_mapping(
    setting: Literal["all", "main", "commands_only"]
) -> np.array:
    with open("data/converted/encoder.pkl", "rb") as f:
        encoder = pkl.load(f)
    match setting:
        case "all":
            mapping = dict(
                zip(range(12), list(encoder.inverse_transform(list(range(12)))))
            )
        case "main":
            mapping = {
                0: "command",
                1: "unknown",
                2: "silence"
            }
        case "commands_only":
            cls_names = list(encoder.inverse_transform(list(range(12))))
            commands_idx = [
                idx
                for idx, cls_name in zip(range(12), cls_names)
                if cls_name not in {"unknown", "silence"}
            ]
            commands_idx_mapping = dict(zip(range(len(commands_idx)), commands_idx))
            mapping = {i: cls_names[commands_idx_mapping[i]] for i in range(len(commands_idx))}
    return mapping

def decode_composed_model_predictions(predictions):
    with open("data/converted/encoder.pkl", "rb") as f:
        encoder = pkl.load(f)
    mapping_commands = get_label_mapping("commands_only")
    mapping_commands.update({-1: "unknown", -2: "silence"})
    predictions = np.array([mapping_commands[prediction] for prediction in predictions])
    return encoder.transform(predictions)
import json
import torch

import torch.utils.data as data_utils


def get_data_loader(sample):
    data = torch.load(f"data/converted/{sample}_data.pt")
    with open(f"data/converted/{sample}_labels.json") as f:
        labels = json.load(f)
    labels = torch.LongTensor(labels)
    data = data.cuda()
    labels = labels.cuda()
    dataset = data_utils.TensorDataset(data, labels)
    loader = data_utils.DataLoader(
        dataset,
        batch_size=64,
        shuffle=(sample == "train"),
    )
    return loader

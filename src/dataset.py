import torch
from pathlib import Path
import numpy as np


def load_from_npy(data_dir, prefix, deeprnn=False):
    npy_datasets = {}
    for dataset in ["X_train", "y_train", "X_test", "y_test"]:
        filename = Path(data_dir, f"{prefix}_{dataset}.npy")
        npy_datasets[dataset] = np.load(filename)

    if deeprnn:
        for dataset in ["X_train_masks", "X_test_masks"]:
            filename = Path(data_dir, f"{prefix}_{dataset}.npy")
            npy_datasets[dataset] = np.load(filename)
    return npy_datasets


def load_test_npy(data_dir, deeprnn=False):
    if deeprnn:
        X_test = Path(data_dir, f"deeprnn_X_test.npy")
        X_test_masks = Path(data_dir, f"deeprnn_X_test_masks.npy")
        y_test = Path(data_dir, f"deeprnn_y_test.npy")
        return np.load(X_test), np.load(X_test_masks), np.load(y_test)
    else:
        X_test = Path(data_dir, f"vitals_X_test.npy")
        y_test = Path(data_dir, f"vitals_y_test.npy")
        return np.load(X_test), np.load(y_test)


class BPDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, idx):
        sequence = torch.tensor(self.inputs[idx], dtype=torch.double)
        label = torch.tensor(self.labels[idx], dtype=torch.double)
        return torch.unsqueeze(sequence, 0), label

    def __len__(self):
        return len(self.labels)


class BPDeepRNNDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, masks, labels):
        self.inputs = inputs
        self.masks = masks
        self.labels = labels

    def __getitem__(self, idx):
        sequence = torch.tensor(self.inputs[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        mask = torch.tensor(self.masks[idx], dtype=torch.int)
        return sequence, mask, label

    def __len__(self):
        return len(self.labels)

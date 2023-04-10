from torch.utils.data import Dataset
import csv
import torch
import numpy as np

class CustomDataset(Dataset):

    def __init__(self, file, transform=None, target_transform=None) -> None:
        with open(file, 'r') as f:
            reader = csv.reader(f)
            raw_data =list(reader)
        self.labels = torch.tensor(np.array([int(raw_data[_][-1]) - 1 for _ in range(len(raw_data))]), dtype=torch.long)
        self.data = torch.tensor(np.array([[int(raw_data[_][idx]) for idx in range(0, len(raw_data[0]) - 1)] for _ in range(len(raw_data))]), dtype=torch.float32)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label
from torch.utils.data import Dataset
import csv
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

class CustomDataset(Dataset):

    def __init__(self, file, transform=None, target_transform=None) -> None:
        scaler = StandardScaler()
        with open(file, 'r') as f:
            reader = csv.reader(f)
            raw_data =list(reader)

        # interested_features = list(set(range(38)) - set(range(3)))
        # interested_features = [3,4,5,6]
        # interested_features = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 25, 26, 27]
        interested_features = list(range(38))

        self.labels = torch.tensor(np.array([int(raw_data[_][-1]) - 1 for _ in range(len(raw_data))]), dtype=torch.long)
        temp_data = torch.tensor(np.array([[int(raw_data[_][idx]) for idx in interested_features] for _ in range(len(raw_data))]),
                                 dtype=torch.float32)
        self.data = torch.empty((len(temp_data), 0), dtype=torch.float32)
        for i in range(len(temp_data[0])):
            if i in [0,1,2,7,8,9,10,11,12,13,25]:
                self.data = torch.cat((self.data, torch.nn.functional.one_hot(temp_data[:, i].to(torch.long))), dim=1)
            elif i in [3,4,5,6,26]:
                scaler.fit(temp_data[:, i:i+1].numpy())
                self.data = torch.cat((self.data, torch.tensor(scaler.transform(temp_data[:, i:i+1].numpy()), dtype=torch.float32)), dim=1)
            else:
                self.data = torch.cat((self.data, temp_data[:, i].unsqueeze(1)), dim=1)
        # self.data = temp_data
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
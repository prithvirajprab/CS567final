from torch.utils.data import Dataset
import csv
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from attribute_mapping import attribute_mapping


class CustomDataset(Dataset):

    def standardizer(self, column_data):
        scaler = StandardScaler()
        scaler.fit(column_data)
        return scaler.transform(column_data)

    def one_hot_encoder(self, column_data, mapping_index, attributemap=True):
        if attributemap:
            for i in range(len(column_data)):
                column_data[i] = attribute_mapping[mapping_index].index(column_data[i])
        column_data = column_data.astype(int)
        return torch.nn.functional.one_hot(torch.tensor(column_data.flatten(), dtype=torch.long))

    def value_converter(self, temp_data):
        data = torch.empty((len(temp_data), 0), dtype=torch.float32)
        
        for i in range(len(temp_data[0])):
            column_data = temp_data[:, i:i+1]
            if i in [7,8,9,10,11,12,13,25]:
                column_data = self.one_hot_encoder(column_data, i, attributemap=True)
            elif i in [0,1,2]:
                column_data = self.one_hot_encoder(column_data, i, attributemap=False)
            elif i in [3,4,5,6,26]:
                column_data = column_data.astype(int)
                column_data = self.standardizer(column_data)
                column_data = torch.tensor(column_data, dtype=torch.float32)
            else:
                column_data = torch.tensor(column_data.astype(int), dtype=torch.long)
            try:
                data = torch.cat((data, column_data), dim=1)
            except Exception as e:
                print(column_data)
                raise e

        return data
    
    def label_converter(self, label_data):
        return torch.tensor(np.array([int(label_data[_][-1]) - 1 for _ in range(len(label_data))]), dtype=torch.long)

    def __init__(self, value_file, label_file=None, transform=None, target_transform=None, encodeFlag=False,
                 removeidslabelflag=False, removeidsvalueflag=False, removebuildingidflag=False,) -> None:

        if label_file:
            with open(label_file, 'r') as f:
                reader = csv.reader(f)
                if removeidslabelflag:
                    label_data =list(reader)[1:]
                else:
                    label_data = list(reader)
            self.labels = self.label_converter(label_data)
        else:
            self.labels = None

        with open(value_file, 'r') as f:
            reader = csv.reader(f)
            if removeidsvalueflag:
                value_data =list(reader)[1:]
            else:
                value_data =list(reader)

        if removebuildingidflag:
            value_data = np.array(value_data)[:, 1:]  # remove the id
        else:
            value_data = np.array(value_data, dtype=np.float32)

        if encodeFlag:
            interested_features = list(range(3))
            self.data = value_data[:, interested_features]
            self.data = self.value_converter(self.data)
        else:
            self.data = value_data

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.labels is not None:
            label = self.labels[idx]
            if self.transform:
                data = self.transform(data)
            if self.target_transform:
                label = self.target_transform(label)
            return data, label
        else:
            return data

    def save_processed_dataset(self, file_path):
        data = self.data.numpy()
        np.savetxt(file_path + '_value.csv', data, delimiter=',', fmt="%.2f")
        if self.labels is not None:
            labels = self.labels.numpy()
            np.savetxt(file_path + '_label.csv', labels, delimiter=',', fmt="%d")
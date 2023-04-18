import csv
import os
import torch
import torch.nn as nn

from neural_net import Net
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.preprocessing import StandardScaler

# Activation functions
elu = nn.ELU # Exponential linear function
softmax = nn.Softmax(dim=1)  # softmax(x_i) = \exp(x_i) / (\sum_j \exp(x_j))
tanh = nn.Tanh()
relu = nn.ReLU()
sigmoid = nn.Sigmoid()

path = "/Users/prithvirajprabhu/Documents/Research projects local/CS 567 final project/Code/CS 567 " \
       "final/Models/Model04_17_16_24_51_120/0.00014554178951086682.pt"
oppath = "/Users/prithvirajprabhu/Documents/Research projects local/CS 567 final project/Code/CS 567 " \
       "final/Outputs/v1.csv"
layersizes = [339,200,125,50,15,3]
acts = [nn.Linear, relu, relu, relu, relu, softmax]
dataset_location = "/Users/prithvirajprabhu/Documents/Research projects local/CS 567 final project/Code/CS 567 " \
             "final/Dataset/"\
          + "TEST-15-49-200.csv"

def get_dimensions(my_list):
    if isinstance(my_list, list):
        return [len(my_list)] + get_dimensions(my_list[0])
    else:
        return []


class CustomDatasetSP(Dataset):

    def __init__(self, file, transform=None, target_transform=None) -> None:
        scaler = StandardScaler()
        with open(file, 'r') as f:
            reader = csv.reader(f)
            raw_data =list(reader)


        print(get_dimensions(raw_data))
        print(raw_data[1])

        self.labels = torch.tensor(np.array([int(raw_data[_][0]) for _ in range(len(raw_data))]), dtype=torch.long)
        temp_data = torch.tensor(np.array([[int(raw_data[_][idx+1]) for idx in range(37)]
                                           for _ in range(2)]), dtype=torch.float32)

        self.data = torch.empty((len(temp_data), 0), dtype=torch.float32)
        for i in range(len(temp_data[0])):
            if i in [0,1,2,7,8,9,10,11,12,13,25]:
                self.data = torch.cat((self.data, torch.nn.functional.one_hot(temp_data[:, i].to(torch.long))), dim=1)
            elif i in [3,4,5,6,26]:
                scaler.fit(temp_data[:, i:i+1].numpy())
                self.data = torch.cat((self.data, torch.tensor(scaler.transform(temp_data[:, i:i+1].numpy()), dtype=torch.float32)), dim=1)
            else:
                self.data = torch.cat((self.data, temp_data[:, i].unsqueeze(1)), dim=1)
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


def single_pass():
	dataloader = DataLoader(CustomDatasetSP(dataset_location), batch_size=1, shuffle=False)

	NetObject = torch.load(path)
	net = Net(layersizes,acts)
	net.load_state_dict(NetObject)

	building_ids = []
	outputs = []

	with torch.no_grad():
		for batch_idx, (X, y) in enumerate(dataloader):
			output = NetObject.forward(X).cpu().detach().numpy()  # output is a length-3 vector of
			# probabilities that sum to 1.
			outputs.extend(np.argmax(output, axis=1))
			building_ids.extend(y.cpu().detach().numpy())

	outputslist = np.vstack(np.array(['building_id', 'damage_grade']),np.hstack((np.array(building_ids).reshape(-1,1), np.array(
		outputs).reshape(-1,1))))

	return outputslist

outputlist = single_pass()

np.savetxt(oppath, outputlist, delimiter=', ')




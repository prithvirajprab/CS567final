import csv
import os
import pdb

import torch
import torch.nn as nn

from neural_net import Net
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.preprocessing import StandardScaler

# Activation functions
from preprocess import CustomDataset

elu = nn.ELU # Exponential linear function
softmax = nn.Softmax(dim=1)  # softmax(x_i) = \exp(x_i) / (\sum_j \exp(x_j))
tanh = nn.Tanh()
relu = nn.ReLU()
sigmoid = nn.Sigmoid()

path = "/Users/prithvirajprabhu/Documents/Research projects local/CS 567 final project/Code/CS 567 final/Models/Model04_19_21_05_20_813/0.00015214361382123013.pt"
oppath = "/Users/prithvirajprabhu/Documents/Research projects local/CS 567 final project/Code/CS 567 " \
       "final/Outputs/v2.csv"
layersizes = [364, 200, 125, 50, 15, 3]
acts = [nn.Linear, relu, relu, relu, relu, softmax]
dataset_location = "/Users/prithvirajprabhu/Documents/Research projects local/CS 567 final project/Code/CS 567 " \
             "final/Dataset/RPMED-31-66-200_Test_values.csv"
original_test_location = "/Users/prithvirajprabhu/Documents/Research projects local/CS 567 final project/Code/CS 567 " \
             "final/Dataset/Richters_Predictor_Modeling_Earthquake_Damage_-_Test_Values.csv"

def get_dimensions(my_list):
    if isinstance(my_list, list):
        return [len(my_list)] + get_dimensions(my_list[0])
    else:
        return []


def single_pass():
	dataset = CustomDataset(dataset_location)
	dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

	net = Net(layersizes,acts)
	net.load_state_dict(torch.load(path))
	net.eval()

	with open(original_test_location, 'r') as f:
		reader = csv.reader(f)
		building_ids = list(reader)[1:]
		building_ids = np.array([building_ids[_][0] for _ in range(len(building_ids))], dtype=np.int64)

	outputs = []
	counter = 0

	for _, x in enumerate(dataloader):
		# pdb.set_trace()
		output = net(x).cpu().detach().numpy() # output is a length-3 vector of
		# probabilities that sum to 1.
		if counter < 20:
			counter += 1
			print(output)
		# pdb.set_trace()
		outputs.extend([int(np.argmax(output))+1])

	# outputslist = np.vstack((np.array(['building_id', 'damage_grade']), np.hstack((np.array(building_ids).reshape(-1,1), np.array(outputs).reshape(-1,1)))))
	outputslist = np.hstack((building_ids.reshape(-1, 1), np.array(outputs).reshape(-1, 1))).tolist()

	return outputslist

outputlist = single_pass()


np.savetxt(oppath, outputlist, delimiter=',')




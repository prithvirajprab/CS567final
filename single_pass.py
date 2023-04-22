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

def to_abs_path(relative_path):
    script_location = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(script_location, relative_path)

path = to_abs_path("Models/Model04_21_18_08_39_398/4.25296298891103e-05.pt")
oppath = to_abs_path("Outputs/v9test.csv")
layersizes = [364, 160, 3]
acts = [nn.Linear, relu, softmax]
dataset_location = to_abs_path("Dataset/RPMED-31-66-200_Test_values.csv")
original_test_location = to_abs_path("Dataset/Richters_Predictor_Modeling_Earthquake_Damage_-_Test_Values.csv")

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

    # net_state_dict = net.state_dict()
    # for layer_name, layer_weights in net_state_dict.items():
    #     # pdb.set_trace()
    #     print(f"{layer_name}: weights={layer_weights}")

    # pdb.set_trace()

    with open(original_test_location, 'r') as f:
        reader = csv.reader(f)
        building_ids = list(reader)[1:]
        building_ids = np.array([building_ids[_][0] for _ in range(len(building_ids))], dtype=np.int64)

    outputs = []
    counter = 0

    for _, x in enumerate(dataloader):
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




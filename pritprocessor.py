from torch.utils.data import Dataset
import csv
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from attribute_mapping import attribute_mapping
import os


# Setting up the dataset
from preprocess import CustomDataset

def to_abs_path(relative_path):
    script_location = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(script_location, relative_path)


value = to_abs_path('Dataset/RPMED-31-66-200.csv')
outfile = to_abs_path('Dataset/TransRPMED-31-66-200')

if __name__ == "__main__":
    dataset = CustomDataset(value, encodeFlag=True)
    dataset.save_processed_dataset(outfile)


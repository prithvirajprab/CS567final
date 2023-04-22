import pdb

from sklearn.decomposition import PCA
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



def PCAfitter(data):
    pca = PCA(n_components=2)
    pca.fit(data)
    params = pca.get_params()
    print(params, pca.explained_variance_ratio_)
    return params

def PCAtransformer(data, params):
    pca = PCA(n_components=2)
    pca.set_params(**params)
    pca_data = pca.transform(data)
    return torch.tensor(pca_data, dtype=torch.float32)

value = to_abs_path('Dataset/RPMED-geodataprePCA.csv')
outfile = to_abs_path('Dataset/postPCAgeo')

if __name__ == "__main__":
    dataset = CustomDataset(value, encodeFlag=False)

    pca = PCA(n_components=2)
    pca.fit(dataset)


    pca_data = pca.transform(dataset)
    print(pca.explained_variance_ratio_)
    # paramsPCA = PCAfitter(dataset)
    # transformeddata = PCAtransformer(dataset, paramsPCA)
    # pdb.set_trace()

    np.savetxt(outfile + '.csv', pca_data, delimiter=',', fmt="%d")


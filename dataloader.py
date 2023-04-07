import csv
import os
import torch
from torch.utils.data import random_split
import numpy as np


def dataloader(filepath, device):

  with open(filepath) as f:
    reader = csv.reader(f)
    data = list(reader)

  sample_size = len(data)
  non_test_size = int(0.8 * sample_size)
  test_size = sample_size - non_test_size
  train_size = int(0.8 * non_test_size)
  valid_size = non_test_size - train_size

  non_test_set, test_set = random_split(data, [non_test_size, test_size], generator=torch.Generator().manual_seed(428))
  train_set, valid_set = random_split(non_test_set, [train_size, valid_size], generator=torch.Generator().manual_seed(1000))

  train_features, valid_features, test_features = ([] for _ in range(3))
  train_labels, valid_labels, test_labels = ([] for _ in range(3))

  for row in range(train_size):
    train_features.append([int(x) for x in train_set[row][:-1]])
    train_labels.append(int(train_set[row][-1]) - 1)

  for row in range(valid_size):
    valid_features.append([int(x) for x in valid_set[row][:-1]])
    valid_labels.append(int(valid_set[row][-1]) - 1)

  for row in range(test_size):
    test_features.append([int(x) for x in test_set[row][:-1]])
    test_labels.append(int(test_set[row][-1]) - 1)

  train_features = torch.tensor(np.array(train_features), dtype=torch.float32, device=device)
  train_labels = torch.tensor(np.array(train_labels), dtype=torch.long, device=device)
  valid_features = torch.tensor(np.array(valid_features), dtype=torch.float32, device=device)
  valid_labels = torch.tensor(np.array(valid_labels), dtype=torch.long, device=device)
  test_features = torch.tensor(np.array(test_features), dtype=torch.float32, device=device)
  test_labels = torch.tensor(np.array(test_labels), dtype=torch.long, device=device)

  return train_features, train_labels, valid_features, valid_labels, test_features, test_labels
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pdb
import pickle as pkl
import os
from ray import tune
import dataloader as dl

from sklearn.metrics import f1_score

from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


class Net(nn.Module):

  def __init__(self, layersizes, acts):
    super(Net, self).__init__()
    self.acts = acts
    self.features = nn.Linear(in_features=layersizes[0], out_features=layersizes[1])
    self.hidden = [nn.Linear(in_features=layersizes[j + 1], out_features=layersizes[j + 2]) for j in range(len(acts) - 3)]
    self.damage = nn.Linear(in_features=layersizes[-2], out_features=layersizes[-1])

  def forward(self, featurerow):

    num_layers = len(self.acts)
    layers = [self.features] + self.hidden + [self.damage]
    a_0 = self.acts[0](featurerow)

    def arch(input, l):
      z_l = layers[l](input)
      a_l = self.acts[l+1](z_l)
      if l < num_layers-2:
        return arch(a_l, l+1)
      else:
        return a_l

    return arch(a_0, 0)


def train(config, checkpoint_dir = None, **kwargs):
  loss_arr = []
  train_acc, valid_acc = [], []

  NetObject = Net(kwargs['layersizes'], kwargs['acts'])
  device="cpu"

  data = dl.dataloader(kwargs["dataset"], device)
  train_features, train_labels, valid_features, valid_labels = data[:4]
  optimizer = optim.Adam(NetObject.parameters(), lr=config['lr'], betas=(0.9, 0.99), eps=1e-08,
                         weight_decay=10 ** -4, amsgrad=False)

  if checkpoint_dir:
    model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
    net.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

  for epoch in range(kwargs['epochs']):
    print(epoch)
    for idx, featurerow in enumerate(train_features):
      featurerow, train_labels[idx] = featurerow.to(device), train_labels[idx].to(device)
      optimizer.zero_grad() # Initializing the gradients to zero
      output = NetObject.forward(featurerow)
      loss = kwargs['criterion'](output, train_labels[idx])
      loss.backward()
      optimizer.step()

    # Measure training and validation accuracy for each epoch
    train_acc_epoch = accuracy(NetObject, train_features, train_labels, **kwargs)
    valid_acc_epoch = accuracy(NetObject, valid_features, valid_labels, **kwargs)
    train_acc.append(train_acc_epoch)
    valid_acc.append(valid_acc_epoch)
    loss_epoch = loss.cpu().detach().numpy()
    loss_arr.append(loss_epoch)
    print("Epoch {}: Loss = {}".format(epoch+1, round(float(loss_epoch), kwargs['precision'])), flush=True, end=', ')
    print("Training Acc = {}, Validation Acc = {}".format(round(train_acc_epoch, kwargs['precision']),
                                                           round(valid_acc_epoch, kwargs['precision'])), flush=True, end=', ')
    torch.save(NetObject, kwargs['mod_filename'])

    # with tune.checkpoint_dir(epoch) as checkpoint_dir:
    #   path = os.path.join(checkpoint_dir, "checkpoint")
    #   torch.save((net.state_dict(), optimizer.state_dict()), path)

    tune.report(loss=loss_epoch, accuracy=train_acc_epoch)

  results = [loss_arr, train_acc, valid_acc]
  with open(kwargs['acc_filename'], "wb") as file:
    pkl.dump(results, file)


def accuracy(NetObject, ds_features, ds_labels, **kwargs):
  #  use a one-hot method of calculating accuracy: (0.2,0.2,0.6) vs (0,0,1)
  l = len(ds_features)
  outputs = []

  with torch.no_grad():
    for idx in range(l):
      output = NetObject.forward(ds_features[idx]).cpu().detach().numpy()
      outputs.append(np.argmax(output))

  return f1_score(outputs, ds_labels.cpu().detach().numpy(), average='micro')






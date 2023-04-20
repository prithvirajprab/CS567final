import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pdb
import pickle as pkl
import os
from ray import tune
from torch.optim.lr_scheduler import StepLR

from preprocess import CustomDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import pdb

from sklearn.metrics import f1_score

from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


class Net(nn.Module):

  def __init__(self, layersizes, acts):
    super(Net, self).__init__()
    self.acts = acts
    self.features = nn.Linear(in_features=layersizes[0], out_features=layersizes[1])
    self.hidden1 = nn.Linear(in_features=layersizes[1], out_features=layersizes[2])
    self.hidden2 = nn.Linear(in_features=layersizes[2], out_features=layersizes[3])
    self.hidden3 = nn.Linear(in_features=layersizes[3], out_features=layersizes[4])
    self.damage = nn.Linear(in_features=layersizes[-2], out_features=layersizes[-1])

  def forward(self, featurerow):
    num_layers = len(self.acts)
    layers = [self.features] + [self.hidden1] + [self.hidden2] + [self.hidden3] + [self.damage]

    def arch(input, l):
      z_l = layers[l](input)
      a_l = self.acts[l+1](z_l)
      if l < num_layers-2:
        return arch(a_l, l+1)
      else:
        return a_l

    return arch(featurerow, 0)


def train(config, checkpoint_dir = None, **kwargs):
  loss_arr = []
  train_acc, valid_acc = [], []

  NetObject = Net(kwargs['layersizes'], kwargs['acts'])
  device="cpu"

  dataset = CustomDataset(kwargs["value"], kwargs['label'], removeidslabelflag=True)
  train_data, valid_data = random_split(
        dataset=dataset,
        lengths=[int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)]
    )
  train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
  valid_dataloader = DataLoader(valid_data, batch_size=config['batch_size'], shuffle=True)

  # optimizer = optim.Adam(NetObject.parameters(), lr=config['lr'], betas=(0.9, 0.99), eps=1e-08,
  #                        weight_decay=10 ** -4, amsgrad=False)
  optimizer = optim.Adam(NetObject.parameters(), lr=config['lr'])  # use_ema, ema_momentum
  scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

  # if checkpoint_dir:
  #   model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
  #   net.load_state_dict(model_state)
  #   optimizer.load_state_dict(optimizer_state)

  for epoch in range(kwargs['epochs']):
    loss_epoch = 0
    batch_count = 0
    # pdb.set_trace()
    for batch_idx, (X, y) in enumerate(train_dataloader):
      X = X.to(device)
      y = y.to(device)
      optimizer.zero_grad() # Initializing the gradients to zero
      output = NetObject(X)
      loss = kwargs['criterion'](output, y)
      # if batch_idx % 1e4 == 0:
      #   print(loss, output[-1], y[-1])
      loss_epoch += loss.item()
      batch_count += 1
      loss.backward()
      optimizer.step()

    # Measure training and validation accuracy for each epoch
    train_acc_epoch = accuracy(NetObject, train_dataloader, **kwargs)
    valid_acc_epoch = accuracy(NetObject, valid_dataloader, **kwargs)
    train_acc.append(train_acc_epoch)
    valid_acc.append(valid_acc_epoch)
    loss_epoch /= batch_count
    loss_arr.append(loss_epoch)
    print("Epoch {}: Loss = {}".format(epoch+1, round(float(loss_epoch), kwargs['precision'])), flush=True, end=', ')
    print("Training Acc = {}, Validation Acc = {}".format(round(train_acc_epoch, kwargs['precision']),
                                                           round(valid_acc_epoch, kwargs['precision'])), flush=True, end=', ')
    # torch.save(NetObject, kwargs['mod_folder']+str(config["lr"])+".pt")

    # with tune.checkpoint_dir(epoch) as checkpoint_dir:
    #   path = os.path.join(checkpoint_dir, "checkpoint")
    #   torch.save((net.state_dict(), optimizer.state_dict()), path)

    os.makedirs(kwargs['mod_folder'], exist_ok=True)
    torch.save(NetObject.state_dict(), os.path.join(kwargs['mod_folder'], str(config["lr"])+".pt"))

    # if train_acc_epoch - valid_acc_epoch > 0.025:
    #   break
    # net_state_dict = NetObject.state_dict()
    # for layer_name, layer_weights in net_state_dict.items():
    #   # pdb.set_trace()
    #   print(f"{layer_name}: weights={layer_weights}")

    tune.report(loss=round(float(loss_epoch), kwargs['precision']), accuracy=round(valid_acc_epoch,
                                                                                   kwargs['precision']))
    scheduler.step()

  results = [loss_arr, train_acc, valid_acc]
  with open(kwargs['acc_filename'], "wb") as file:
    pkl.dump(results, file)

def accuracy(NetObject, data_loader, **kwargs):
  #  I find the class with "maximum probability" and then compare that to the given labels using the microf1 metric .
  ground_truth = []
  outputs = []

  with torch.no_grad():
    for batch_idx, (X, y) in enumerate(data_loader):
      output = NetObject.forward(X).cpu().detach().numpy()  # output is a length-3 vector of
      # probabilities that sum to 1.
      outputs.extend(np.argmax(output, axis=1))
      ground_truth.extend(y.cpu().detach().numpy())

  return f1_score(ground_truth, outputs, average='micro')


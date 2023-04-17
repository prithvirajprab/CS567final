import dataloader as dl
from neural_net import Net, train
import sys
import datetime
import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from ray import tune

from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# Activation functions
elu = nn.ELU # Exponential linear function
softmax = nn.Softmax(dim=1)  # softmax(x_i) = \exp(x_i) / (\sum_j \exp(x_j))
tanh = nn.Tanh()
relu = nn.ReLU()
sigmoid = nn.Sigmoid()


# Setting up the device and dataset
# device = "cpu"
dataset = '/Users/prithvirajprabhu/Documents/Research projects local/CS 567 final project/Code/CS 567 ' \
          'final/Dataset/RPMED-15-49-200.csv'
checkpoint_dir = None
num_samples = 10
max_num_epochs = 250
gpus_per_trial = 0

# Defining the architecture
layersizes = [339,200,125,50,15,3]
acts = [nn.Linear, relu, relu, relu, relu, softmax]
NetObject = Net(layersizes, acts)

# if torch.cuda.is_available():
#     device = "cuda:0"
#     if torch.cuda.device_count() > 1:
#         net = nn.DataParallel(net)
# net.to(device)

# Filenames
timestamp = str(datetime.datetime.now())[5:23].replace(":", "_").replace(".", "_").replace(" ", "_").replace("-", "_")
print(timestamp, layersizes, acts, max_num_epochs)
mod_filename = "/Users/prithvirajprabhu/Documents/Research projects local/CS 567 final project/Code/CS 567 final/Models/model_"+timestamp+".pt"
acc_filename = "/Users/prithvirajprabhu/Documents/Research projects local/CS 567 final project/Code/CS 567 final/Accuracy/acc_"+timestamp+".pkl"

# Hyperparameters
kwargs = {'epochs': max_num_epochs,
          'dataset': dataset,
          'momentum': 0.9,
          'layersizes': layersizes,
          'acts': acts,
          'precision': 4,
          'criterion': nn.CrossEntropyLoss(),
          'mod_filename': mod_filename,
          'acc_filename': acc_filename,
          'checkpoint_dir': checkpoint_dir}

# Ray tune wrappers
config = {
	"lr": tune.loguniform(5e-5,2e-4),
    'batch_size': tune.qrandint(lower=20, upper=25, q=2)
}
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=max_num_epochs,
    grace_period=40,
    reduction_factor=1.3)
reporter = CLIReporter(
    # parameter_columns=["l1", "l2", "lr", "batch_size"],
    metric_columns=["loss", "accuracy", "training_iteration"],
    max_report_frequency=120)

# Main training function
result = tune.run(
    partial(train, **kwargs),
    resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter, verbose=1)

best_trial = result.get_best_trial("loss", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(
    best_trial.last_result["loss"]))
print("Best trial final validation accuracy: {}".format(
    best_trial.last_result["accuracy"]))


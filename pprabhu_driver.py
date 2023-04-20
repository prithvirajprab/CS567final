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
import os

# Activation functions
elu = nn.ELU # Exponential linear function
softmax = nn.Softmax(dim=1)  # softmax(x_i) = \exp(x_i) / (\sum_j \exp(x_j))
tanh = nn.Tanh()
relu = nn.ReLU()
sigmoid = nn.Sigmoid()

# convert the relative path to absolute path, so no need to modify the path every time!
def to_abs_path(relative_path):
    script_location = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(script_location, relative_path)

# Setting up the dataset
value = to_abs_path('Dataset/RPMED-31-66-200_Train_values.csv')

# set this to None or empty string when only do testing
label = to_abs_path('Dataset/Richters_Predictor_Modeling_Earthquake_Damage_-_Train_Labels.csv')

checkpoint_dir = None
num_samples = 10
max_num_epochs = 100
gpus_per_trial = 0

# Defining the architecture
layersizes = [364, 200, 125, 60, 15, 3]
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
# mod_filename = "/Users/prithvirajprabhu/Documents/Research projects local/CS 567 final project/Code/CS 567 final/Models/model_"+timestamp+".pt"
acc_filename = to_abs_path("Accuracy/acc_"+timestamp+".pkl")
mod_folder = to_abs_path("Models/Model"+timestamp)


# Hyperparameters
kwargs = {'epochs': max_num_epochs,
          'value': value,
          'label': label,
          'momentum': 0.9,
          'layersizes': layersizes,
          'acts': acts,
          'precision': 4,
          'criterion': nn.CrossEntropyLoss(),
          'mod_folder': mod_folder,
          'acc_filename': acc_filename,
          'checkpoint_dir': checkpoint_dir}

# Ray tune wrappers
config = {
	"lr": tune.loguniform(7e-5,3e-4),
    'batch_size': tune.qrandint(lower=16, upper=25, q=2)
}
scheduler = ASHAScheduler(
    metric="accuracy",
    mode="max",
    max_t=max_num_epochs,
    grace_period=50,
    reduction_factor=1.15)
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

best_trial = result.get_best_trial("accuracy", "max", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(
    best_trial.last_result["loss"]))
print("Best trial final validation accuracy: {}".format(
    best_trial.last_result["accuracy"]))


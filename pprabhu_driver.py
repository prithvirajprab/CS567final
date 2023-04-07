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
softmax = nn.Softmax(dim=0) # softmax(x_i) = \exp(x_i) / (\sum_j \exp(x_j))
tanh = nn.Tanh()
relu = nn.ReLU()
sigmoid = nn.Sigmoid()

# Setting up the device and dataset
# device = "cpu"
dataset = '/Users/prithvirajprabhu/Documents/Research projects local/CS 567 final project/Code/CS 567 final/Dataset/RPMED-Traincombined.csv'
# dataset = '/Documents/Research projects local/CS 567 final project/Code/CS 567 final/Dataset/RPMED-Traincombined.csv'
# data = dl.dataloader(dataset, device)
checkpoint_dir = None
num_samples = 1
max_num_epochs = 50
gpus_per_trial = 0

# Defining the architecture
#layersizes = [6, 20, 30, 35, 45, 14]
layersizes = [38,100,10,3]
acts = [relu, tanh, tanh, softmax]
NetObject = Net(layersizes, acts)

# if torch.cuda.is_available():
#     device = "cuda:0"
#     if torch.cuda.device_count() > 1:
#         net = nn.DataParallel(net)
# net.to(device)

num_epochs = 10
learning_rate = 0.001
# learning_rate_final_epoch =0.0001 # must be less than learning_rate
trials_at_end = 35
trials_offset = 10

# Filenames
# print(sys.argv)
# if sys.argv[1] == "n":
timestamp = str(datetime.datetime.now())[5:23].replace(":", "_").replace(".", "_").replace(" ", "_").replace("-", "_")
print(timestamp, layersizes, acts, num_epochs, learning_rate, trials_at_end)
# elif sys.argv[1] == "e":
#   timestamp = sys.argv[2]
# mod_filename = "/project/tbrun_769/qdec/models/model_"+timestamp+".pt"
# acc_filename = "/project/tbrun_769/qdec/models/acc_"+timestamp+".pkl"

mod_filename = "/Users/prithvirajprabhu/Documents/Research projects local/CS 567 final project/Code/CS 567 final/Models/model_"+timestamp+".pt"
acc_filename = "/Users/prithvirajprabhu/Documents/Research projects local/CS 567 final project/Code/CS 567 final/Accuracy/acc_"+timestamp+".pkl"

# Hyperparameters
kwargs = {'epochs': num_epochs,
          'dataset': dataset,
          # 'learningRate': learning_rate,
          # 'learningLast': learning_rate_final_epoch,
          'momentum': 0.9,
          'layersizes': layersizes,
          'acts': acts,
          'precision': 5,
	  	  # 'trials_offset':trials_offset,
          'criterion': nn.CrossEntropyLoss(),
          'mod_filename': mod_filename,
          'acc_filename': acc_filename,
          'checkpoint_dir': checkpoint_dir}

# Ray tune wrappers
config = {
	"lr": tune.loguniform(1e-4, 1e-3)
}
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=max_num_epochs,
    grace_period=1,
    reduction_factor=1.5)
reporter = CLIReporter(
    # parameter_columns=["l1", "l2", "lr", "batch_size"],
    metric_columns=["loss", "accuracy", "training_iteration"],
    max_report_frequency=60)
result = tune.run(
    partial(train, **kwargs),
    resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter)

best_trial = result.get_best_trial("loss", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(
    best_trial.last_result["loss"]))
print("Best trial final validation accuracy: {}".format(
    best_trial.last_result["accuracy"]))

#train(QuantumDecoderNet, checkpoint_dir, device, *data[:4], **kwargs)



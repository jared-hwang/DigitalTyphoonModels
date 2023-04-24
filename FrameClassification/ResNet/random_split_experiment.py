import datetime
import os

import torch
import torch.nn as nn
from pathlib import Path

from train_utils import train, validate
from model import ResidualBlock, ResNet
from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataroot', required=True, type=str, help='path to the root data directory')
parser.add_argument('--savepath', required=True, type=str, help='path to the directory to save the models and logs')
parser.add_argument('--loaddata', default=False, type=bool, help='path to the root data directory')
args = parser.parse_args()
dataroot = Path(args.dataroot)
data_path = dataroot
save_path = Path(args.savepath)
if args.loaddata:
    load_data = 'all_data'
else:
    load_data = False

# data_path = Path.home() / 'data'
images_path = str(data_path / 'image') + '/'
track_path = str(data_path / 'track') + '/'
metadata_path = str(data_path / 'metadata.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 83
torch.manual_seed(SEED)

# Create the model
num_classes = 7
num_epochs = 3
batch_size = 32
learning_rate = 0.01

# Models
models = []
optimizers = []
for i in range(5):
    models.append(ResNet(ResidualBlock, [3, 4, 6, 3]).to(device))
    optimizers.append(torch.optim.Adam(models[-1].parameters(), lr=learning_rate, weight_decay = 0.001))

# Loss
criterion = nn.CrossEntropyLoss()

# Open the dataset
# Open the dataset
dataset = DigitalTyphoonDataset(str(images_path),
                                str(track_path),
                                str(metadata_path),
                                load_data_into_memory=load_data,
                                verbose=False)

for i, model in enumerate(models):
    train_set, test_set = dataset.random_split([0.8, 0.2], split_by='frame')
    train_indices = train_set.indices
    
    model = model.to(device)
    train_log_string = train(model, dataset, train_set, optimizers[i], criterion, num_epochs, batch_size, device, save_path)
    val_loss, val_acc, val_f1_result = validate(model, dataset, test_set, criterion, device, save_path)
    train_log_string += f"Validation: \n \t loss: {val_loss} \n \t acc: {val_acc} \n \t val_f1: {val_f1_result}"

    curr_time_str = str(datetime.datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))
    torch.save(model.state_dict(), str(save_path / 'saved_models' / f'resnet_weights_{curr_time_str}'))

    file1 = open("random_split_results.txt", "a")  # append mode
    file1.write(train_log_string)
    file1.close()


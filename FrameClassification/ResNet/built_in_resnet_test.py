import datetime
import os

import torch
import torch.nn as nn
from pathlib import Path

from train_utils import train, validate
from torch.utils.data import DataLoader
from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataroot', required=True, type=str, help='path to the root data directory')
parser.add_argument('--savepath', required=True, type=str, help='path to the directory to save the models and logs')
parser.add_argument('--loaddata', default=False, type=bool, help='path to the root data directory')
parser.add_argument('--small', default=False, type=bool, help='Bool to use small or full dataset')
args = parser.parse_args()
dataroot = Path(args.dataroot)
data_path = dataroot
save_path = Path(args.savepath)
if args.loaddata:
    load_data = 'all_data'
else:
    load_data = False

use_small = args.small

# data_path = Path.home() / 'data'
images_path = str(data_path / 'image') + '/'
track_path = str(data_path / 'track') + '/'
metadata_path = str(data_path / 'metadata.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 83
torch.manual_seed(SEED)

# Create the model
num_classes = 7
num_epochs = 1
batch_size = 16
learning_rate = 0.01

start_time_str = str(datetime.datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(in_features=512, out_features=8, bias=True)

# Trying max pool
# model.avgpool = torch.nn.AdaptiveMaxPool2d(output_size=(1, 1))

model = model.to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

# Open the dataset
dataset = DigitalTyphoonDataset(str(images_path),
                                str(track_path),
                                str(metadata_path),
                                'grade',
                                load_data_into_memory=load_data,
                                verbose=False)

if use_small:
    train_set, test_set, _ = dataset.random_split([0.01, 0.01, 0.98], split_by='frame')
else:
    train_set, test_set = dataset.random_split([0.8, 0.2], split_by='frame')

trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
train_log_string = train(model, trainloader, optimizer, criterion, num_epochs, device, save_path)

curr_time_str = str(datetime.datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))
testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
val_log_string = validate(model, testloader, criterion, device, curr_time_str, save_path)
train_log_string += val_log_string

torch.save(model.state_dict(), str(save_path / 'saved_models' / f'built_in_resnet_weights_{curr_time_str}.pt'))

with open(str(save_path / 'logs' / f'built_in_resnet_log_{curr_time_str}'), 'w') as writer:
    writer.write(train_log_string)
    writer.write(f'\n start time: {start_time_str}')




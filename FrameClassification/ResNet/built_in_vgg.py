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

SEED = 83
torch.manual_seed(SEED)

# Create the model
num_classes = 7
num_epochs = 4
batch_size = 16
learning_rate = 0.01

model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
model.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)

#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
#model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
model = model.to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

# Open the dataset
dataset = DigitalTyphoonDataset(str(images_path),
                                str(track_path),
                                str(metadata_path),
                                load_data_into_memory=load_data,
                                verbose=False)

# Open the dataset
# dataset = DigitalTyphoonDataset("/Users/jaredhwang/PycharmProjects/typhoonDataset/tests/test_data_files/image/",
#                                 "/Users/jaredhwang/PycharmProjects/typhoonDataset/tests/test_data_files/track/",
#                                 "/Users/jaredhwang/PycharmProjects/typhoonDataset/tests/test_data_files/metadata.json",
#                                 verbose=False)

train_set, test_set = dataset.random_split([0.8, 0.2], split_by='frame')
train_indices = train_set.indices

train_log_string = train(model, dataset, train_set, optimizer, criterion, num_epochs, batch_size, device, save_path)
val_loss, val_acc, val_f1_result = validate(model, dataset, test_set, criterion, device, save_path)
train_log_string += f"Validation: \n \t loss: {val_loss} \n \t acc: {val_acc} \n \t val_f1: {val_f1_result}"

curr_time_str = str(datetime.datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))
torch.save(model.state_dict(), str(save_path / 'saved_models' / f'built_in_resnet_weights_{curr_time_str}'))

with open(str(save_path / 'logs' / f'built_in_resnet_log_{curr_time_str}'), 'w') as writer:
    writer.write(train_log_string)





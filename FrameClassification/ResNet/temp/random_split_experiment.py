import datetime

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


def train_one_resnet_model(split_by, save_path, split_lengths=None, num_epochs=30, batch_size=16, learning_rate=0.01):
    if split_lengths is None:
        split_lengths = [0.8, 0.2]

    split_save_path = save_path / (split_by + '_logs')

    start_time_str = str(datetime.datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(in_features=512, out_features=8, bias=True)

    model = model.to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    def image_filter(image):
        return image.grade() < 7

    # Open the dataset
    dataset = DigitalTyphoonDataset(str(images_path),
                                    str(track_path),
                                    str(metadata_path),
                                    'grade',
                                    load_data_into_memory=load_data,
                                    filter_func=image_filter,
                                    verbose=False)

    if use_small:
        train_set, test_set, _ = dataset.random_split([0.01, 0.01, 0.98], split_by=split_by)
    else:
        train_set, test_set = dataset.random_split(split_lengths, split_by=split_by)

    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

    train_log_string = f'Start time: {start_time_str} \n' \
                       f'Num epochs: {num_epochs} \n' \
                       f'Batch size: {batch_size} \n' \
                       f'Learning rate: {learning_rate} \n ' \
                       f'Split by: {split_by} \n '

    train_log_string += train(model, trainloader, optimizer, criterion, num_epochs, device, split_save_path)

    curr_time_str = str(datetime.datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
    val_log_string = validate(model, testloader, criterion, device, curr_time_str, split_save_path, num_classes=5)
    train_log_string += val_log_string

    torch.save(model.state_dict(), str(split_save_path / 'saved_models' / f'built_in_resnet_weights_{curr_time_str}.pt'))

    with open(str(split_save_path / 'logs' / f'log_{curr_time_str}.txt'), 'w') as writer:
        writer.write(train_log_string)

# Run sequence splits
# num_loops = 10
# for i in range(num_loops):
#     train_one_resnet_model('sequence', save_path,  split_lengths=[0.8, 0.2], num_epochs=30, batch_size=16, learning_rate=0.01)

# Run year splits
num_loops = 10
for i in range(num_loops):
    train_one_resnet_model('year', save_path,  split_lengths=[0.8, 0.2], num_epochs=40, batch_size=16, learning_rate=0.1)


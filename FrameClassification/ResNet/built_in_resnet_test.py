import datetime
import argparse

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

from train_utils import train, validate, Logger
from torch.utils.data import DataLoader
from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataroot', required=True, type=str, help='path to the root data directory')
parser.add_argument('--savepath', required=True, type=str, help='path to the directory to save the models and logs')
parser.add_argument('--loaddata', default=False, type=bool, help='path to the root data directory')
parser.add_argument('--small', default=False, type=bool, help='Bool to use small or full dataset')
parser.add_argument('--split_by', default='frame', type=str, help='How to split the dataset')
parser.add_argument('--autostop', default=False, type=bool, help='If the training should be run until max epochs or until it stops converging')
parser.add_argument('--maxepochs', default=100, type=int)
args = parser.parse_args()
dataroot = Path(args.dataroot)
data_path = dataroot
save_path = Path(args.savepath)
save_path = save_path / (args.split_by + '_logs')
load_data = 'all_data' if args.loaddata else False
use_small = args.small
logger = Logger()
start_time_str = str(datetime.datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))

# data_path = Path.home() / 'data'
images_path = str(data_path / 'image') + '/'
track_path = str(data_path / 'track') + '/'
metadata_path = str(data_path / 'metadata.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model
num_epochs = 1
batch_size = 16
learning_rate = 0.0001
num_workers = 8

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(in_features=512, out_features=7, bias=True)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


# Open the dataset
def image_filter(image):
    return image.grade() < 7

transform_func = None
def transform_func(image_ray):
    image_ray = np.clip(image_ray, 150, 350)
    image_ray = (image_ray - 150) / (350 - 150)
    image_ray = torch.Tensor(image_ray)
    image_ray = torch.reshape(image_ray, [1, 1, image_ray.size()[0], image_ray.size()[1]])
    image_ray = nn.functional.interpolate(image_ray, size=(224, 224), mode='bilinear', align_corners=False)
    image_ray = torch.reshape(image_ray, [image_ray.size()[2], image_ray.size()[3]])
    image_ray = image_ray.numpy()
    return image_ray

dataset = DigitalTyphoonDataset(str(images_path),
                                str(track_path),
                                str(metadata_path),
                                'grade',
                                load_data_into_memory=load_data,
                                filter_func=image_filter,
                                transform_func=transform_func,
                                verbose=False)

if use_small:
    train_set, test_set, _ = dataset.random_split([0.001, 0.001, 0.998], split_by=args.split_by)
else:
    train_set, test_set = dataset.random_split([0.8, 0.2], split_by=args.split_by)

trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

logger.log(f'Start time: {start_time_str} \n '
           f'Num/max epochs: {num_epochs} \n '
           f'Batch size: {batch_size} \n'
           f'Learning rate: {learning_rate} \n '
           f'Split by: {args.split_by} \n'
           f'Autostop: {args.autostop} \n'
           f'Normalized: {True if transform_func is not None else False}')

logger.log_json('meta', 'start_time', start_time_str)
logger.log_json('meta', 'num_max_epochs', num_epochs)
logger.log_json('meta', 'batch_size', batch_size)
logger.log_json('meta', 'lr', learning_rate)
logger.log_json('meta', 'split_by', args.split_by)
logger.log_json('meta', 'autostop', args.autostop)
logger.log_json('meta', 'standardized', True if transform_func is not None else False)


initial_val_loss, initial_val_acc = validate(model, testloader, criterion, device, start_time_str, save_path, num_classes=5, log_results=-1, logger=logger)
logger.print(f"\n\t Val loss: {initial_val_loss} \t Val acc: {initial_val_acc}")
logger.log(f"Initial Val Loss: {initial_val_loss} \t Initial Val Acc: {initial_val_acc}")

if args.autostop:
    train(model, trainloader, testloader, optimizer, criterion, args.maxepochs, device, save_path, autostop=True, autostop_parameters=(3, 0.3), logger=logger)
else:
    train(model, trainloader, testloader, optimizer, criterion, num_epochs, device, save_path, autostop=None, logger=logger)

validate(model, testloader, criterion, device, start_time_str, save_path, num_classes=5, log_results=num_epochs-1, logger=logger)

torch.save(model.state_dict(), str(save_path / 'saved_models' / f'built_in_resnet_weights_{start_time_str}.pt'))
logger.write(str(save_path / 'logs' / f'log_{start_time_str}.txt'))
logger.write_json(str(save_path / 'logs' / f'log_{start_time_str}.json'))



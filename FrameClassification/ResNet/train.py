import datetime
import torch
import torch.nn as nn
from pathlib import Path

from train_utils import train, validate
from model import ResidualBlock, ResNet
from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataroot', required=True, type=str, help='path to the root data directory')
args = parser.parse_args()
dataroot = Path(args.dataroot)
data_path = dataroot

# data_path = Path.home() / 'data'
images_path = str(data_path / 'image') + '/'
track_path = str(data_path / 'track') + '/'
metadata_path = str(data_path / 'metadata.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 83
torch.manual_seed(SEED)

# Create the model
num_classes = 7
num_epochs = 10
batch_size = 16
learning_rate = 0.01
model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

# Open the dataset
dataset = DigitalTyphoonDataset(str(images_path),
                                str(track_path),
                                str(metadata_path),
                                verbose=False)

# Open the dataset
# dataset = DigitalTyphoonDataset("/Users/jaredhwang/PycharmProjects/typhoonDataset/tests/test_data_files/image/",
#                                 "/Users/jaredhwang/PycharmProjects/typhoonDataset/tests/test_data_files/track/",
#                                 "/Users/jaredhwang/PycharmProjects/typhoonDataset/tests/test_data_files/metadata.json",
#                                 verbose=False)

train_set, test_set = dataset.random_split([0.8, 0.2], split_by='frame')
train_indices = train_set.indices

train(model, dataset, train_set, optimizer, criterion, num_epochs, batch_size, device)
validate(model, dataset, test_set, criterion, device)

torch.save(model.state_dict(), f'saved_models/resnet_{datetime.datetime.now().strftime("%d/%m/%Y-%H.%M.%S")}')



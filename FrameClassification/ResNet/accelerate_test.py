import datetime

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from train_utils import train, validate
from torch.utils.data import DataLoader
from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset

from accelerate import Accelerator

accelerator = Accelerator()

# Create the model
num_epochs = 100
batch_size = 16
learning_rate = 0.001
num_workers = 0
split_by = 'sequence'

start_time_str = str(datetime.datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(in_features=512, out_features=8, bias=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

data_path = Path('/data')
save_path = Path('/DigitalTyphoonModels/FrameClassification/ResNet/accelerate_logs/')
images_path = str(data_path / 'image') + '/'
track_path = str(data_path / 'track') + '/'
metadata_path = str(data_path / 'metadata.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def image_filter(image):
    return image.grade() < 7

# Open the dataset
dataset = DigitalTyphoonDataset(str(images_path),
                                str(track_path),
                                str(metadata_path),
                                'grade',
                                load_data_into_memory=True,
                                filter_func=image_filter,
                                verbose=False)

train_set, test_set = dataset.random_split([0.8, 0.2], split_by=split_by)
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

train_log_string = f'Start time: {start_time_str} \n' \
                   f'Num/max epochs: {num_epochs} \n' \
                   f'Batch size: {batch_size} \n' \
                   f'Learning rate: {learning_rate} \n ' \
                   f'Split by: {split_by} \n' \
                   f'Autostop: {False} \n' \
                   f'Accelerate: True'

curr_time_str = str(datetime.datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))

model, optimizer, training_dataloader = accelerator.prepare(
    model, optimizer, trainloader
)

for epoch in range(num_epochs):
    for batch_num, data in enumerate(tqdm(trainloader)):
        optimizer.zero_grad()
        inputs, targets = data
        inputs = torch.reshape(inputs, [inputs.size()[0], 1, inputs.size()[1], inputs.size()[2]])

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()

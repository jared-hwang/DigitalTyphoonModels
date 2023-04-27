import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import alexnet
from tqdm import tqdm

from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device used: ', device)

# Import the database in a dataset_obj
print('importing dataset...')
dataset_obj = DigitalTyphoonDataset("/home/dataset/image/", 
                                    "/home/dataset/track/", 
                                    "/home/dataset/metadata.json",
                                    split_dataset_by='frame',
                                    load_data_into_memory='all_data',
                                    get_images_by_sequence=False,
                                    ignore_list=[],
                                    verbose=False)

# Split Data
g1 = torch.Generator().manual_seed(83)
train, test = dataset_obj.random_split([0.80, 0.20], split_by='frame', generator=g1)
print('%d images loaded in the train_set'% len(train.indices))

# Define parameters
n = 50000 #len(train)
batch_size = 16
mean = 269.6207
std = 36.0843

# Instantiate the network and define the loss function and optimizer
net = alexnet(num_classes=8)
net.features[0]= nn.Conv2d(1,64,kernel_size=11,stride=4,padding=2)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the network
print('Start training')
for epoch in range(100):
    print('Epoch: ', epoch)
    running_loss = 0.0
    with tqdm(range(0, n, batch_size), dynamic_ncols=True) as pbar:
        for i in pbar:
            inputs = ((dataset_obj.images_as_tensor(train.indices[i: i + batch_size]) - mean) / std).to(device)
            assert not torch.any(torch.isnan(inputs))
            inputs = inputs.reshape(inputs.shape[0], 1, 512, 512).to(device)
            labels = dataset_obj.labels_as_tensor(train.indices[i: i + batch_size], 'grade').to(device)
    
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if loss.isnan():
                print("Error: exploding gradient issue at image ", i)
                break

            if i % 2000 == 0:
                running_loss = 0.0
            else:
                pbar.set_postfix({'loss': running_loss/(i%2000)})

    # Save the current model
    PATH = 'model_vuillod/models_25_04/net_%d_tmp%d.pth'% (n, epoch)
    print(PATH)
    torch.save(net.state_dict(), PATH)
    print("model saved with %d epochs"% epoch)

print('Finished Training')

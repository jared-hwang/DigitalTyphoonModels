import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import alexnet, vit_b_16
from tqdm import tqdm

from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset

def filter(image):
    return (image.grade() < 7)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device used: ', device)

# Import the database in a dataset_obj
print('importing dataset...')
dataset_obj = DigitalTyphoonDataset("/home/dataset/image/",
                                    "/home/dataset/track/", 
                                    "/home/dataset/metadata.json",
                                    'grade',
                                    split_dataset_by='frame',
                                    get_images_by_sequence=False,
                                    load_data_into_memory='all_data',
                                    ignore_list=[],
                                    filter_func=filter,
                                    verbose=False)

# Split Data
g1 = torch.Generator().manual_seed(83)
train_part = 0.8
train, test = dataset_obj.random_split([train_part, 1-train_part], split_by='frame', generator=g1)
print('%d images loaded in the train_set'% len(train.indices))

train_loader = torch.utils.data.DataLoader(train, batch_size=16,
                                           shuffle=True, num_workers=8)

# Define mean and std for standardization
mean = 269.5767
std = 34.3959

# Instantiate the network and define the loss function and optimizer
net = vit_b_16(num_classes=8)
patch_size = 16
net.conv_proj = nn.Conv2d(in_channels=1, out_channels=768, kernel_size=patch_size, stride=patch_size)


# net.features[0]= nn.Conv2d(1,64,kernel_size=11,stride=4,padding=2)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the network
print('Start training')
for epoch in range(100):
    print('Epoch: ', epoch)
    running_loss = 0.0
    with tqdm(train_loader, dynamic_ncols=True) as pbar:
        for i, data in enumerate(pbar, 0):
            if i==500:
                break
            inputs, labels = data
            inputs = ((inputs - mean) / std).float()
            assert not torch.any(torch.isnan(inputs))
            inputs = inputs.reshape(inputs.shape[0], 1, 512, 512).to(device)
            inputs = nn.functional.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)
            labels = labels.to(device)

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

    if loss.isnan():
        break
    # Save the current model
    PATH = 'net_%0.2f_500_tmp%d.pth'% (train_part, epoch)
    print(PATH)
    torch.save(net.state_dict(), PATH)
    print("model saved with %d epochs and trained with %d%% of the images"% (epoch, train_part*100))

print('Finished Training')
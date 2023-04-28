import torch
import torch.nn as nn
from torchvision.models import alexnet
from tqdm import tqdm

from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('device used: ', device)

def filter(image):
    return (image.grade() < 7)

# Import the database in a dataset_obj
print('importing dataset...')
dataset_obj = DigitalTyphoonDataset("/home/dataset/image/", 
                                    "/home/dataset/track/", 
                                    "/home/dataset/metadata.json",
                                    'grade',
                                    split_dataset_by='frame',
                                    load_data_into_memory='all_data',
                                    get_images_by_sequence=False,
                                    ignore_list=[],
                                    filter_func=filter,
                                    verbose=False)

# Split Data
g1 = torch.Generator().manual_seed(83)
train_part = 0.8
train, test = dataset_obj.random_split([train_part, 1 - train_part], split_by='frame', generator=g1)
print('%d images loaded in the test_set'% len(test.indices))


test_loader = torch.utils.data.DataLoader(train, batch_size=16,
                                           shuffle=True, num_workers=8)

# Define mean and std for standardization
mean = 269.5767
std = 34.3959

# Instantiate the network
net = alexnet(num_classes=8)
net.features[0]= nn.Conv2d(1,64,kernel_size=11,stride=4,padding=2)
net = net.to(device)

all_confusion_matrices = torch.zeros(100, 8, 8, dtype=int)

# Test the network
print('Start testing')
for epoch in range(100):
    PATH = 'net_0.80_tmp%d.pth'% epoch
    net.load_state_dict(torch.load(PATH))
    net.eval()

    correct = 0
    total = 0

    print('testing ', PATH)
    with tqdm(test_loader, dynamic_ncols=True) as pbar:
        for i, data in enumerate(pbar, 0):
            inputs, labels = data
            inputs = ((inputs - mean) / std).float()
            assert not torch.any(torch.isnan(inputs))
            inputs = inputs.reshape(inputs.shape[0], 1, 512, 512).to(device)
            labels = labels.to(device)

            # Prediction
            with torch.no_grad():
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Confusion matrix computing
            for j in range(len(labels)):
                true_label = int(labels[j].item())
                pred_label = predicted[j].item()
                all_confusion_matrices[epoch, true_label, pred_label] += 1

    # Results
    print('Accuracy of the network with %d epochs on %d %% of test images: %d %%' %
        (epoch, 1-train_part, 100 * correct / total))
    print('confusion matrix :\n', all_confusion_matrices[epoch].int())

    # Saves
    f = open("accuracies.txt", "a")
    f.write(str(epoch) + ": " + str(100 * correct / total) + '\n')
    f.close()
    # torch.save(all_confusion_matrices, 'all_confusion_matrices.pt')
    
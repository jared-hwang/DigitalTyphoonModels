import torch
import torch.nn as nn
from torchvision.models import alexnet
from tqdm import tqdm
import os.path

from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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
test, train = dataset_obj.random_split([0.80, 0.20], split_by='frame', generator=g1)

# Define parameters
n = 37851 #len(test)
mean = 269.6207
std = 36.0843
batch_size = 16
print('number of images in test_set: ', n)

# Instantiate the network
net = alexnet(num_classes=8)
net.features[0]= nn.Conv2d(1,64,kernel_size=11,stride=4,padding=2)
net = net.to(device)

# if os.path.isfile('all_confusion_matrices.pt'):
#     all_confusion_matrices = torch.load('all_confusion_matrices.pt')
# else:
all_confusion_matrices = torch.zeros(100, 8, 8, dtype=int)
    # torch.save(all_confusion_matrices, 'all_confusion_matrices.pt')

# Test the network
for epoch in range(100):
    PATH = 'model_vuillod/models_25_04/net_50000_tmp%d.pth'% epoch
    net.load_state_dict(torch.load(PATH))
    net.eval()

    correct = 0
    total = 0
    accuracies = []

    print('testing ', PATH)
    for i in tqdm(range(0, n, batch_size)):
        test_image_tensor = (dataset_obj.images_as_tensor(test.indices[i: i + batch_size]) - mean) / std
        test_image_tensor = test_image_tensor.reshape(test_image_tensor.shape[0], 1, 512, 512).to(device)
        test_label_tensor = dataset_obj.labels_as_tensor(test.indices[i: i + batch_size], 'grade').to(device)

        # Prediction
        with torch.no_grad():
            images, labels = test_image_tensor, test_label_tensor.long()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Confusion matrix computing
        for j in range(len(labels)):
            true_label = labels[j]
            pred_label = predicted[j]
            all_confusion_matrices[epoch, true_label, pred_label] += 1

    # Results
    print('Accuracy of the network with %d epochs on %d test images: %d %%' %
        (epoch, n, 100 * correct / total))
    print('confusion matrix :\n', all_confusion_matrices[epoch].int())

    # Saves
    f = open("accuracies_25_04_trainset.txt", "a")
    f.write(str(epoch) + ": " + str(100 * correct / total) + '\n')
    f.close()

    # torch.save(all_confusion_matrices, 'all_confusion_matrices.pt')
    accuracies.append(100 * correct / total)
    print(accuracies)

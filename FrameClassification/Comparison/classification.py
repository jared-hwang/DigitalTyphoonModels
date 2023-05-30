import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import alexnet, vgg16_bn, resnet18, vit_b_16

from tqdm import tqdm
import datetime
import os

import matplotlib.pyplot as plt
import seaborn as sn

def init_model(model_name):
    if model_name == "alexnet":
        model = alexnet(num_classes=8)
        model.features[0]= nn.Conv2d(1,64,kernel_size=11,stride=4,padding=2)
        return model
    elif model_name == "vgg16_bn":
        model = vgg16_bn(num_classes=8)
        model.features[0]= nn.Conv2d(1,64,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        model.features[-1]=nn.AdaptiveMaxPool2d(7*7)
        model.classifier[-1]=nn.Linear(in_features = 4096, out_features=8, bias = True)
        return model
    elif model_name == "resnet18":
        model = resnet18(num_classes=7)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=8, bias=True)
        return model  
    elif model_name == "vit_b_16":
        model = vit_b_16(num_classes=8)
        patch_size = 16
        model.conv_proj = nn.Conv2d(in_channels=1, out_channels=768, kernel_size=patch_size, stride=patch_size)
        return model
    else:
        raise Exception("Model name unknown:" + model_name)


def cm_compute(confusion_matrix, labels, predicted):
    for j in range(len(labels)):
                  true_label = int(labels[j].item())
                  pred_label = predicted[j].item()
                  confusion_matrix[true_label, pred_label] += 1
    return confusion_matrix

def compute_acc_from_cm(cm):
    true_pos = torch.tensor([cm[i,i] for i in range(5)]).sum()
    all = cm.sum()
    acc = true_pos / all *100
    return acc

def input_transform(inputs, mean, std, resolution):
    inputs = ((inputs - mean) / std).float()
    assert not torch.any(torch.isnan(inputs))
    inputs = inputs.reshape(inputs.shape[0], 1, 512, 512)
    inputs = nn.functional.interpolate(inputs, size=resolution, mode='bilinear', align_corners=False)
    return inputs

def add_and_save(stat, tensor_save_path, epoch):
    save_tensor = torch.load(tensor_save_path)
    save_tensor[epoch] = stat
    torch.save(save_tensor, tensor_save_path)

def train_and_test(dataset_obj, device, 
                   model_name, weights,
                   SEED, train_part,
                   n_epochs, 
                   mean, std, 
                   learning_rate, momentum,
                   result_path):

    # Split Data
    g1 = torch.Generator().manual_seed(SEED)
    train, test = dataset_obj.random_split([train_part, 1-train_part], generator=g1)

    train_loader = torch.utils.data.DataLoader(train, batch_size=16,
                                                shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=16,
                                            shuffle=True, num_workers=8)

    # Instantiate the network and define the loss function and optimizer
    model = init_model(model_name).to(device)
    last_saved = -1
    #   model.load_state_dict(torch.load(result_path + model_name + '/' + 'net0.80_epoch%d.pth'%(last_i_saved) ))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    result_path = result_path + model_name + '/'
    if os.path.exists(result_path):
        raise Exception('Warning, result folder already exist and may contain results')
    else:
        os.makedirs(result_path)

    # Train the network
    print('%d images loaded in the train_set'% len(train.indices))
    print('\nStart training ' + model_name)
    start_time_str = str(datetime.datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))

    train_log_string = f'Start time: {start_time_str} \n' \
                       f'Image dir: {dataset_obj.image_dir} \n' \
                       f'Track dir: {dataset_obj.track_dir} \n' \
                       f'Device: {device} \n' \
                       f'Model: {model_name} \n' \
                       f'Weights: {weights} \n' \
                       f'SEED: {SEED} \n' \
                       f'train part: {train_part} \n' \
                       f'Split by: {dataset_obj.split_dataset_by} \n' \
                       f'Num epochs: {n_epochs} \n' \
                       f'Batch size: 16 \n' \
                       f'Mean: {mean} \n' \
                       f'Std: {std} \n' \
                       f'Learning rate: {learning_rate} \n' \
                       f'Momentum: {momentum} \n' \
                       f'Train set length: {len(train)} \n' \
                       f'Test set length: {len(test)}'

    print(train_log_string)
    with open(str(result_path + f'log_{start_time_str}.txt'), 'w') as writer:
        writer.write(train_log_string)
        writer.close()

    # Init save tensors
    train_losses = torch.zeros(n_epochs, dtype=float)
    torch.save(train_losses, result_path + "train_losses.pt")
    test_losses = torch.zeros(n_epochs, dtype=float)
    torch.save(test_losses, result_path + "test_losses.pt")
    accuracies = torch.zeros(n_epochs, dtype=float)
    torch.save(accuracies, result_path + "accuracies.pt")

    for epoch in range(last_saved +1, n_epochs):
        print('Epoch: ', epoch)
        PATH = result_path + 'net%0.2f_epoch%d.pth'% (train_part, epoch)
        print('Training ' + PATH)

        train_loss_sum = 0.0
        with tqdm(train_loader, ncols=100) as pbar:
            pbar.set_description('train ' + model_name + '_ep' + str(epoch))
            for i, data in enumerate(pbar, 0):
                inputs, labels = data
                inputs = torch.clamp(inputs, 150, 350)
                inputs = input_transform(inputs, mean, std, (224, 224))
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()

                if loss.isnan():
                    print("Error: exploding gradient issue at image ", i)
                    break
                
                pbar.set_postfix({'loss': train_loss_sum/(i+1)})

        if loss.isnan():
            break
        # Save the current model and train loss
        torch.save(model.state_dict(), PATH)
        if epoch != 0: os.remove(result_path + 'net%0.2f_epoch%d.pth'% (train_part, epoch-1))
        print("model saved with %d epochs and trained with %d%% of the images"% (epoch, train_part*100))
        add_and_save(train_loss_sum/len(train), result_path + "train_losses.pt", epoch)

        # TESTING
        print('Testing ', PATH)
        confusion_matrix = torch.zeros(8, 8, dtype=int)
        test_loss_sum = 0

        with tqdm(test_loader, dynamic_ncols=True) as pbar:
            pbar.set_description('test ' + model_name + '_ep' + str(epoch))
            for i, data in enumerate(pbar, 0):
                inputs, labels = data
                inputs = input_transform(inputs, mean, std, (224, 224)).to(device)
                labels = labels.to(device)

                # Prediction
                with torch.no_grad():
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels.long())
                    test_loss_sum += loss.item()
                confusion_matrix = cm_compute(confusion_matrix, labels, predicted)
                
                pbar.set_postfix({'loss': train_loss_sum/(i+1)})

        # Results
        confusion_matrix = confusion_matrix[2:7, 2:7]
        acc = compute_acc_from_cm(confusion_matrix)
        print('Accuracy of the network with %d epochs on %d%% of test images: %d %%' %
            (epoch, (1-train_part)*100, acc))

        # Save test_loss, accuracy, heatmap and confusion_matrix tensor
        add_and_save(test_loss_sum/len(test), result_path + "test_losses.pt", epoch)
        add_and_save(acc, result_path + "accuracies.pt", epoch)
        
        torch.save(confusion_matrix, result_path + 'confusion_matrix_%d.pt'%(epoch))
        if epoch != 0: os.remove(result_path + 'confusion_matrix_%d.pt'%(epoch - 1))
        
        sn.heatmap(confusion_matrix, annot=True, fmt='g', xticklabels=range(2,7), yticklabels=range(2,7))
        plt.savefig(result_path + f'confusion_matrix_{epoch}.png')
        plt.close()
        if epoch != 0: os.remove(result_path + f'confusion_matrix_{epoch-1}.png')

    print('Finished Training')

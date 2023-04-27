import torch
import datetime
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from torchvision import datasets, transforms, models


def train(model, trainloader, optimizer, criterion,
          epochs, device, savepath):
    batches_per_epoch = len(trainloader)
    log_string = ''
    model.train()

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        train_running_loss = 0.0
        train_running_correct = 0
        total = 0
        for batch_num, data in enumerate(tqdm(trainloader)):
            images, labels = data
            images, labels = torch.Tensor(images).float(), torch.Tensor(labels).long()
            images = torch.reshape(images, [images.size()[0], 1, images.size()[1], images.size()[2]])
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(images)

            # Calculate the loss
            loss = criterion(predictions, labels)
            train_running_loss += loss.item()

            # Calculate the accuracy.
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            train_running_correct += (predicted == labels).sum().item()

            # backward pass
            loss.backward()
            # update weights
            optimizer.step()


        # Loss and accuracy for the complete epoch.
        epoch_loss = train_running_loss / batches_per_epoch
        epoch_acc = 100. * (train_running_correct / total)
        print(f"\t Avg batch Loss: {epoch_loss}")
        print(f"\t Accuracy: {epoch_acc}%")
        log_string += f"Epoch {epoch + 1} \n \t loss: {epoch_loss} \n \t acc: {epoch_acc} \n"

        # Checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, str(savepath / 'saved_models' / f'model_checkpoint_epoch{epoch + 1}.pt'))

    return log_string


def validate(model, testloader, criterion, device, timestring, savepath):
    print('Validation')
    model.eval()
    valid_running_loss = 0.0

    truth_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch_num, data in enumerate(tqdm(testloader)):
            images, labels = data
            images, labels = images.float(), labels.long()
            images = torch.reshape(images, [images.size()[0], 1, images.size()[1], images.size()[2]]).to(device)
            labels = labels.to(device)

            # Forward pass
            predictions = model(images)

            predicted_label = torch.argmax(predictions, 1).to('cpu')
            predicted_labels.extend(predicted_label)
            truth_labels.extend(labels.to('cpu'))

            # Calculate the loss
            loss = criterion(predictions, labels)
            valid_running_loss += loss.item()


        # Loss and accuracy for the complete epoch.
        epoch_loss = valid_running_loss
        micro_f1_result = f1_score(truth_labels, predicted_labels, average='micro')
        macro_f1_result = f1_score(truth_labels, predicted_labels, average='macro')
        weighted_f1_result = f1_score(truth_labels, predicted_labels, average='weighted')
        accuracy = 100 * accuracy_score(truth_labels, predicted_labels)

        num_classes = 6
        # Build confusion matrix
        cf_matrix = confusion_matrix(truth_labels, predicted_labels)
        df_cm = pd.DataFrame(cf_matrix, index=[i for i in range(num_classes)],
                             columns=[i for i in range(num_classes)])

        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True, fmt='g')
        plt.savefig(str(savepath / 'logs' / f'resnet_confusion_matrix_{timestring}.png'))

        # Save normalized confusion matrix
        plt.clf()
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in range(num_classes)],
                             columns=[i for i in range(num_classes)])
        sn.heatmap(df_cm, annot=True, fmt='g')
        plt.savefig(str(savepath / 'logs' / f'resnet_confusion_matrix_norm_{timestring}.png'))

        log_string = ''
        log_string += f"Validation: \n \t loss: {epoch_loss} \n \t acc: {accuracy} \n \t micro_f1: {micro_f1_result} " \
                      f"\n \t macro_f1: {macro_f1_result}\n \t weighted_f1: {weighted_f1_result}"

        print(f"\t Loss: {epoch_loss}")
        print(f"\t Accuracy: {accuracy}%")
        print(f"\t Micro F1 Score: {micro_f1_result}")
        print(f"\t Macro average F1 Score: {macro_f1_result}")
        print(f"\t Weighted average F1 Score: {weighted_f1_result}")
        return log_string

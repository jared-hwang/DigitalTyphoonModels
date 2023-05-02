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


def train_one_epoch(model, trainloader, optimizer, criterion, epoch, device, savepath):
    batches_per_epoch = len(trainloader)
    num_train_samples = len(trainloader.dataset)
    model.train()
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
    epoch_loss = train_running_loss / num_train_samples
    epoch_acc = 100. * (train_running_correct / total)

    return epoch_loss, epoch_acc


def train(model, trainloader, testloader, optimizer, criterion, max_epochs,
          device, savepath, autostop=(3, 0.03)):
    log_string = ''
    epoch_losses = []
    validation_losses = []

    model.train()

    if autostop is not None:
        early_stopper = EarlyStopper(patience=autostop[0], min_delta=autostop[1])
    else:
        early_stopper = None

    for epoch in np.arange(max_epochs):
        epoch_loss, epoch_acc = train_one_epoch(model, trainloader, optimizer, criterion, epoch, device, savepath)
        print(f"\t Avg Sample Loss: {epoch_loss}")
        print(f"\t Accuracy: {epoch_acc}%")
        log_string += f"Epoch {epoch + 1} \n \t loss: {epoch_loss} \n \t acc: {epoch_acc} \n"
        epoch_losses.append(epoch_loss)

        validation_loss = validate(model, testloader, criterion, device, None, savepath, save_results=False, num_classes=5)
        print(f'\t Validation loss: {validation_loss}')
        log_string += f'Validation loss: {validation_loss} \n'
        validation_losses.append(validation_loss)

        # Checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, str(savepath / 'saved_models' / f'model_checkpoint_epoch{epoch + 1}.pt'))

        if early_stopper is not None:
            if early_stopper.early_stop(validation_loss):
                break

        print("Epoch losses: ", epoch_losses)
        print("Validation losses: ", validation_losses)

    return log_string

def validate(model, testloader, criterion, device, timestring, savepath, save_results=False, num_classes=5):
    print('Validation')
    model.eval()
    valid_running_loss = 0.0
    num_validate_samples = len(testloader.dataset)

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
        epoch_loss = valid_running_loss / num_validate_samples
        micro_f1_result = f1_score(truth_labels, predicted_labels, average='micro')
        macro_f1_result = f1_score(truth_labels, predicted_labels, average='macro')
        weighted_f1_result = f1_score(truth_labels, predicted_labels, average='weighted')
        accuracy = 100 * accuracy_score(truth_labels, predicted_labels)

        if not save_results:
            return epoch_loss
        else:
            # Build confusion matrix
            cf_matrix = confusion_matrix(truth_labels, predicted_labels)
            df_cm = pd.DataFrame(cf_matrix, index=[i+2 for i in range(num_classes)],
                                 columns=[i+2 for i in range(num_classes)])

            plt.figure(figsize=(12, 7))
            sn.heatmap(df_cm, annot=True, fmt='g')
            plt.savefig(str(savepath / 'logs' / f'resnet_confusion_matrix_{timestring}.png'))

            # Save normalized confusion matrix
            plt.clf()
            df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i+2 for i in range(num_classes)],
                                 columns=[i+2 for i in range(num_classes)])
            sn.heatmap(df_cm, annot=True, fmt='g')
            plt.savefig(str(savepath / 'logs' / f'resnet_confusion_matrix_norm_{timestring}.png'))

            log_string = ''
            log_string += f"Validation: \n \t loss: {epoch_loss} \n \t acc: {accuracy} \n \t micro_f1: {micro_f1_result} " \
                          f"\n \t macro_f1: {macro_f1_result}\n \t weighted_f1: {weighted_f1_result}"

            print(f"\t Avg Sample Loss: {epoch_loss}")
            print(f"\t Accuracy: {accuracy}%")
            print(f"\t Micro F1 Score: {micro_f1_result}")
            print(f"\t Macro average F1 Score: {macro_f1_result}")
            print(f"\t Weighted average F1 Score: {weighted_f1_result}")
            return log_string


class EarlyStopper:
    def __init__(self, patience=2, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

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


def train_one_epoch(model, trainloader, optimizer, criterion, epoch, device, logger=None):
    logger.print(f"Epoch: {epoch+1}")
    batches_per_epoch = len(trainloader)

    model.train()

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

    return epoch_loss, epoch_acc


def train(model, trainloader, testloader, optimizer, criterion, max_epochs,
          device, savepath, autostop=False, autostop_parameters=(3, 0.03), logger=None):
    log_string = ''
    epoch_losses = []
    validation_losses = []
    validation_accs = []

    early_stopper = EarlyStopper(patience=autostop_parameters[0], min_delta=autostop_parameters[1]) if autostop else None

    for epoch in np.arange(max_epochs):

        # Train one epoch
        epoch_loss, epoch_acc = train_one_epoch(model, trainloader, optimizer, criterion, epoch, device, logger=logger)
        logger.print(f"\t Avg batch Loss: {epoch_loss} \n \t Accuracy: {epoch_acc}%")
        logger.log(f"\n Epoch {epoch + 1} \n \t loss: {epoch_loss} \n \t acc: {epoch_acc}")
        epoch_losses.append(epoch_loss)

        # Run evaluation
        validation_loss, validation_acc = validate(model, testloader, criterion, device, None, savepath, save_results=False, num_classes=5, logger=logger)
        logger.print(f'\t Validation loss: {validation_loss} \n \t Validation acc: {validation_acc}')
        logger.log(f'\t Validation loss: {validation_loss} \n \t Validation acc: {validation_acc}')
        validation_losses.append(validation_loss)
        validation_accs.append(validation_acc)

        # Checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, str(savepath / 'saved_models' / f'model_checkpoint_epoch{epoch + 1}.pt'))

        if early_stopper is not None:
            if early_stopper.early_stop(validation_loss):
                break

    logger.print(f'Epoch Losses: {epoch_losses} \n Validation losses: {validation_losses} \n Validation accuracies: {validation_accs}')
    logger.log(f'Epoch Losses: {epoch_losses} \n Validation losses: {validation_losses} \n Validation accuracies: {validation_accs}')
    return log_string

def validate(model, testloader, criterion, device, timestring, savepath, save_results=False, num_classes=5, logger=None):
    logger.print('Validation')

    model.eval()
    valid_running_loss = 0.0
    num_batches = len(testloader)

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
        epoch_loss = valid_running_loss / num_batches
        micro_f1_result = f1_score(truth_labels, predicted_labels, average='micro')
        macro_f1_result = f1_score(truth_labels, predicted_labels, average='macro')
        weighted_f1_result = f1_score(truth_labels, predicted_labels, average='weighted')
        accuracy = 100 * accuracy_score(truth_labels, predicted_labels)

        if save_results:
            cm = build_confusion_matrix(predicted_labels, truth_labels, num_classes, timestring, savepath)
            validation_results_string = f'\t Avg Batch Loss: {epoch_loss} \n' \
                                        f'\t Accuracy: {accuracy}% \n' \
                                        f'Micro F1: {micro_f1_result} \n' \
                                        f'Macro F1: {macro_f1_result} \n ' \
                                        f'Weighted F1: {weighted_f1_result}'
            logger.log('Validation: \n \t ' + validation_results_string)
            logger.log(f'Confusion Matrix: \n \t {cm}')
            logger.print(validation_results_string)

        return epoch_loss, accuracy

def build_confusion_matrix(predicted_labels, truth_labels, num_classes, timestring, savepath):
    # Build confusion matrix
    cf_matrix = confusion_matrix(truth_labels, predicted_labels)
    df_cm = pd.DataFrame(cf_matrix, index=[i + 2 for i in range(num_classes)],
                         columns=[i + 2 for i in range(num_classes)])

    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.savefig(str(savepath / 'logs' / f'resnet_confusion_matrix_{timestring}.png'))

    # Save normalized confusion matrix
    plt.clf()
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i + 2 for i in range(num_classes)],
                         columns=[i + 2 for i in range(num_classes)])
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.savefig(str(savepath / 'logs' / f'resnet_confusion_matrix_norm_{timestring}.png'))

    return cf_matrix


class Logger:
    def __init__(self):
        self.log_string = ''

    def print(self, print_str):
        print(print_str)

    def log(self, log_str):
        self.log_string += log_str + '\n'

    def write(self, path):
        with open(path, 'w') as writer:
            writer.write(self.log_string)

class EarlyStopper:
    def __init__(self, patience=2, min_delta=0.):
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

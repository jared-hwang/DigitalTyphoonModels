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


def train(model, dataset, train_indices, optimizer, criterion,
          epochs, batch_size, device, savepath):
    batches_per_epoch = len(train_indices) // batch_size
    log_string = ''
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}")
        train_running_loss = 0.0
        train_running_correct = 0
        total = 0
       
        model.train()
	# for i in range(batches_per_epoch):
        for i in tqdm(range(batches_per_epoch), total=batches_per_epoch):
            start = i * batch_size
            batch_indices = train_indices.indices[start:start + batch_size]

            # take a batch
            Xbatch = dataset.images_as_tensor(batch_indices)
            Xbatch = torch.reshape(Xbatch, [batch_size, 1, 512, 512]).to(device)
            ybatch = dataset.labels_as_tensor(batch_indices, 'grade').long().to(device)

            optimizer.zero_grad()
            # forward pass
            y_pred = model(Xbatch)

            # Calculate the loss.
            loss = criterion(y_pred, ybatch)
            train_running_loss += loss.item()

            # Calculate the accuracy.
            _, predicted = torch.max(y_pred.data, 1)
            total += ybatch.size(0)
            train_running_correct += (predicted == ybatch).sum().item()

            # backward pass
            loss.backward()
            # update weights
            optimizer.step()

        # Loss and accuracy for the complete epoch.
        epoch_loss = train_running_loss / (batches_per_epoch)
        epoch_acc = 100. * (train_running_correct / len(train_indices))
        print(f"\t Avg batch Loss: {epoch_loss}")
        print(f"\t Accuracy: {epoch_acc}%")
        log_string += f"Epoch {epoch+1} \n \t loss: {epoch_loss} \n \t acc: {epoch_acc} \n"

        # Checkpoint
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, str(savepath / 'saved_models' / f'model_checkpoint_epoch{epoch+1}.pt'))
    return log_string

# # Validation function.
def validate(model, dataset, testloader, criterion, device, savepath):
    model.eval()
    print('Validation')
    predicted_labels = []
    truth_labels = []
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image = torch.reshape(torch.Tensor(testloader[i].image()), [1, 1, 512, 512]).to(device)
            label = torch.Tensor([testloader[i].grade()]).long().to(device)

            # Forward pass.
            outputs = model(image)

            
            truth_labels.append(float(label[0].to('cpu')))
            predicted_label = torch.argmax(outputs).to('cpu')
            predicted_labels.append(float(predicted_label))

            # Calculate the loss.
            loss = criterion(outputs, label)
            valid_running_loss += loss.item()

            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == label).sum().item()

    
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss
    epoch_acc = 100. * (valid_running_correct / len(testloader))
    f1_result = f1_score(truth_labels, predicted_labels, average='weighted')
    accuracy = 100*accuracy_score(truth_labels, predicted_labels)
    print(f'epoch acc: {epoch_acc}, sklearn acc: {accuracy}')
    epoch_acc = accuracy

    num_classes = 6
    # Build confusion matrix
    cf_matrix = confusion_matrix(truth_labels, predicted_labels)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in range(num_classes)],
                         columns=[i for i in range(num_classes)])    
     
    curr_time_str = str(datetime.datetime.now().strftime("%d_%m_%Y-%H.%M.%S"))
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.savefig(str(savepath / 'logs' / f'resnet_confusion_matrix_{curr_time_str}.png'))
    
    #normalize confusion mat
    plt.clf()
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in range(num_classes)],
                         columns=[i for i in range(num_classes)])
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.savefig(str(savepath / 'logs' / f'resnet_confusion_matrix_norm_{curr_time_str}.png'))

    print(f"\t Loss: {epoch_loss}")
    print(f"\t Accuracy: {epoch_acc}%")
    print(f"\t Weighted average F1 Score: {f1_result}")
    return epoch_loss, epoch_acc, f1_result


# def validate(model, dataset, test_indices, device):
#     # evaluate trained model with test set
#     with torch.no_grad():
#         y_pred = model(X)
#     accuracy = (y_pred.round() == y).float().mean()
#     print("Accuracy {:.2f}".format(accuracy * 100))


# def train(model, trainloader, optimizer, criterion, device):
#     model.train()
#     print('Training')
#     train_running_loss = 0.0
#     train_running_correct = 0
#     counter = 0
#     for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
#         counter += 1
#         image = torch.from_numpy(data.image())
#         labels = torch.FloatTensor([data.grade()])
#
#         # image, labels = data
#         image = image.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
#
#         # Forward pass.
#         outputs = model(image)
#         print(outputs)
#         exit()
#
#         # Calculate the loss.
#         loss = criterion(outputs, labels)
#         train_running_loss += loss.item()
#
#         # Calculate the accuracy.
#         _, preds = torch.max(outputs.data, 1)
#         train_running_correct += (preds == labels).sum().item()
#
#         # Backpropagation
#         loss.backward()
#         # Update the weights.
#         optimizer.step()
#
#     # Loss and accuracy for the complete epoch.
#     epoch_loss = train_running_loss / counter
#     epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
#     return epoch_loss, epoch_acc
#
#
# # Validation function.
# def validate(model, testloader, criterion, device):
#     model.eval()
#     print('Validation')
#     valid_running_loss = 0.0
#     valid_running_correct = 0
#     counter = 0
#     with torch.no_grad():
#         for i, data in tqdm(enumerate(testloader), total=len(testloader)):
#             counter += 1
#
#             image, labels = data
#             image = image.to(device)
#             labels = labels.to(device)
#             # Forward pass.
#             outputs = model(image)
#
#             # Calculate the loss.
#             loss = criterion(outputs, labels)
#             valid_running_loss += loss.item()
#
#             # Calculate the accuracy.
#             _, preds = torch.max(outputs.data, 1)
#             valid_running_correct += (preds == labels).sum().item()
#
#     # Loss and accuracy for the complete epoch.
#     epoch_loss = valid_running_loss / counter
#     epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
#     return epoch_loss, epoch_acc


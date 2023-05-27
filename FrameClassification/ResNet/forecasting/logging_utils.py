import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import json


class Logger:
    def __init__(self):
        self.log_string = ''
        self.log_json_dict = {}

    def print(self, print_str):
        print(print_str)

    def log_json(self, epoch, key, val):
        if epoch not in self.log_json_dict:
            self.log_json_dict[epoch] = {}
        self.log_json_dict[epoch][key] = val

    def log_json_pairs(self, epoch, pairs):
        for pair in pairs:
            self.log_json(epoch, pair[0], pair[1])

    def log(self, log_str):
        self.log_string += log_str + '\n'

    def log_json_and_txt(self, epoch, key, val):
        self.log_json(epoch, key, val)
        self.log(f'{key}: {val}')

    def log_json_and_txt_pairs(self, epoch, pairs):
        self.log_json_pairs(epoch, pairs)
        for key, val in pairs:
            self.log(f'{key}: {val}')

    def write(self, path):
        with open(path, 'w') as writer:
            writer.write(self.log_string)

    def write_json(self, path):
        print(self.log_json_dict)
        with open(path, 'w') as outfile:
            json.dump(self.log_json_dict, outfile, indent=4)


def build_confusion_matrix(predicted_labels, truth_labels):
    # Build confusion matrix
    cf_matrix = confusion_matrix(truth_labels, predicted_labels)
    return cf_matrix


def save_confusion_matrix(cf_matrix, savepath):
    num_classes = len(cf_matrix)

    df_cm = pd.DataFrame(cf_matrix, index=[i + 2 for i in range(num_classes)],
                         columns=[i + 2 for i in range(num_classes)])

    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    # plt.savefig(str(savepath / 'logs' / f'resnet_confusion_matrix_{timestring}.png'))
    plt.savefig(str(savepath))

    # Save normalized confusion matrix
    plt.clf()
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i + 2 for i in range(num_classes)],
                         columns=[i + 2 for i in range(num_classes)])
    sn.heatmap(df_cm, annot=True, fmt='g')
    # plt.savefig(str(savepath / 'logs' / f'resnet_confusion_matrix_norm_{timestring}.png'))
    plt.savefig(str(savepath))

    return cf_matrix

def read_json_file(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)
    return data

def save_loss_plot_from_json(json_path, save_path):
    data = read_json_file(json_path)
    epoch_losses = data['meta']['epoch_losses']
    test_losses = data['meta']['validation_losses']
    lr = data['meta']['lr']
    batch_size = data['meta']['batch_size']
    split_by = data['meta']['split_by']
    standardized = data['meta']['standardized']
    weights = data['meta']['weights']

    plt.plot(epoch_losses, label='train')
    plt.plot(test_losses, label='test')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Avg batch loss')
    plt.title(f'Test loss per epoch: lr={lr} batch_size={batch_size} standardized={standardized}, split={split_by}, weights={weights}')
    plt.savefig(save_path)


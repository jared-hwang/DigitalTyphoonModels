#/bin/env/python3

import torch
import matplotlib.pyplot as plt
import seaborn as sn
import os
import numpy as np

def compute_acc(cm):
    true_pos = torch.tensor([cm[i,i] for i in range(5)]).sum()
    all = cm.sum()
    acc = true_pos / all *100
    return acc

def compute_all_acc(matrices_iterator):
    acc_list = []
    for cm in matrices_iterator:
        acc_list.append(compute_acc(cm))
    return acc_list

def acc_graph(acc_list, save_path):
    plt.plot(range(len(acc_list)), acc_list)
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.savefig(save_path + 'accuracy_graph.png')
    plt.close()

def generate_heatmaps(cm_iterator, save_path):
    for (i, cm) in enumerate(cm_iterator):
        sn.heatmap(cm, annot=True, fmt='g', xticklabels=range(2,7), yticklabels=range(2,7))
        plt.xlabel('predicted grade')
        plt.ylabel('actual grade')
        plt.savefig(save_path + '/' + f'confusion_matrix_{i}.png')
        plt.close()

def generate_table(conf_matrix):
    classes = range(conf_matrix.shape[0])
    precision = []
    recall = []
    f1_score = []
    support = []
    
    for i in classes:
        tp = conf_matrix[i, i]
        fp = torch.sum(conf_matrix[:, i]) - tp
        fn = torch.sum(conf_matrix[i, :]) - tp
        tn = torch.sum(conf_matrix) - tp - fp - fn
        
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        
        precision.append(p.item())
        recall.append(r.item())
        f1_score.append(f1.item())
        support.append(torch.sum(conf_matrix[i, :]).item())

    data = np.vstack((precision, recall, f1_score, support)).T
    table = plt.table(cellText=data, rowLabels=classes, colLabels=['Precision', 'Recall', 'F1-score', 'Support'], cellLoc='center', loc='center')
    table.set_fontsize(14)
    table.scale(1, 2)
    plt.axis('off')
    plt.savefig('stat0.png')

def matrices_iterate(path):
    i = 0
    cm_path = cm_str(path, i)
    while os.path.exists(cm_path):
        cm = torch.load(cm_path)
        yield(cm)
        i+=1
        cm_path = cm_str(path, i)

def cm_str(model_path, i):
    return model_path + 'confusion_matrix_' + str(i) + '.pt'

def matrix_from_path(cm_path):
    return torch.load(cm_path)

def crop_zeros(arr):
    """Find the index of the last non-zero element and crop the array"""
    last_non_zero_index = len(arr) - 1
    while last_non_zero_index >= 0 and arr[last_non_zero_index] == 0:
        last_non_zero_index -= 1
    return arr[:last_non_zero_index + 1]

def path_to_plot(result_path, model_name, data_name):
    """plot a graph from a data array"""
    data_array = torch.load(result_path + model_name + data_name)
    data_array = crop_zeros(data_array)
    plt.plot(range(len(data_array)), data_array, label=model_name + data_name[:-3])
    plt.xlabel('epoch')
    plt.ylabel(data_name[:-3])
    plt.title(result_path)
    plt.legend()


def main():
    result_path = '/home/results_vuillod/models_05_17/'
    model_names = ['vgg16_bn/', 'vit_b_16/', 'resnet18/']
    for model_name in model_names:
        path_to_plot(result_path, model_name, 'accuracies.pt')
        plt.savefig(result_path + model_name + 'accuracies_graph.png')
        plt.close()

        path_to_plot(result_path, model_name, 'test_losses.pt')
        path_to_plot(result_path, model_name, 'train_losses.pt')
        
        plt.savefig(result_path + model_name + 'losses_graph.png')
        plt.close()


if __name__ == '__main__':
    main()

#/bin/env/python3

import torch
import matplotlib.pyplot as plt
import seaborn as sn
import os
from sklearn.metrics import precision_recall_fscore_support
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

def main():
    resnet_path = '/home/results_vuillod/models_05_03/resnet/'
    # resnet_acc_list = compute_all_acc(resnet_path + 'confusion_matrix_', 100)
    # acc_graph(resnet_acc_list, resnet_path)
    # all_resnet_cm = matrices_iterate('/home/results_vuillod/models_05_03/resnet/')
    # generate_heatmaps(all_resnet_cm, '/home/results_vuillod/tmp/')
    cm = matrix_from_path(cm_str(resnet_path, 99))
    generate_table(cm)

    pass
    

if __name__ == '__main__':
    main()

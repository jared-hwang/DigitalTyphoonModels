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

def main():
    # vgg_path = '/home/results_vuillod/models_05_03/vgg/'
    # all_vgg_cm = matrices_iterate(vgg_path)
    # vgg_acc_list = compute_all_acc(all_vgg_cm)
    # plt.plot(range(len(vgg_acc_list)), vgg_acc_list, label='vgg')

    resnet_path = '/home/results_vuillod/models_05_09/resnet18/'
    all_resnet_cm = matrices_iterate(resnet_path)
    resnet_acc_list = compute_all_acc(all_resnet_cm)
    plt.plot(range(len(resnet_acc_list)), resnet_acc_list, label='resnet')

    # vit_path = '/home/results_vuillod/models_05_03/transformer/'
    # all_vit_cm = matrices_iterate(vit_path)
    # vit_acc_list = compute_all_acc(all_vit_cm)
    # plt.plot(range(len(vit_acc_list)), vit_acc_list, label='vit')
    
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.savefig('accuracy_graph.png')
    plt.close()
    

if __name__ == '__main__':
    main()

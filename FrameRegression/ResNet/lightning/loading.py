import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from FrameDatamodule import TyphoonDataModule
import config

from datetime import datetime
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pathlib import Path
import numpy as np

from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset

import random




def load(type,dataset,batch_size,num_workers):
    train, test = [],[]
    
    if type==0 :
        with open('save/old_train.txt','r') as file:
            train_id=[line for line in file]
        with open('save/old_val.txt','r') as file:
            test_id =[line for line in file]
    if type==1 :
        with open('save/recent_train.txt','r') as file:
            train_id=[line for line in file]
        with open('save/recent_val.txt','r') as file:
            test_id =[line for line in file]
    if type==2 :
        with open('save/now_train.txt','r') as file:
            train_id=[line for line in file]
        with open('save/now_val.txt','r') as file:
            test_id =[line for line in file]
    if type==3 :        
        with open('save/now_train.txt','r') as file:
            train_id1=[line for line in file]
        with open('save/now_val.txt','r') as file:
            test_id1 =[line for line in file]            
        with open('save/recent_train.txt','r') as file:
            train_id2=[line for line in file]
        with open('save/recent_val.txt','r') as file:
            test_id2 =[line for line in file]
        train_id = train_id1 +train_id2
        test_id = test_id1+ test_id2
            
    train_id = [x.replace('\n', '') for x in train_id]    
    test_id = [x.replace('\n','') for x in test_id]
    train = DataLoader(dataset.images_from_sequences(train_id),batch_size= batch_size,num_workers=num_workers,shuffle=True)
    test = DataLoader(dataset.images_from_sequences(test_id),batch_size= batch_size,num_workers=num_workers,shuffle=False)
    
    
    return train, test

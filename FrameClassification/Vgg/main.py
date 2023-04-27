import torch
from torch import nn
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset

from model import *
from training import *
from testing import *
from saving import *




#variables :
train_size = 0.8
test_size = 0.1
validation_size = 0.1
num_class = 6

if(train_size+test_size+validation_size !=1) : print(f"WARNING : SUM OF SIZES != 1 !!")

split = 'frame'

batch_size = 16
epochs = 5
path_to_save = f"./digtyp/models/model_{epochs}_{batch_size}"




#loading data base
dataset_obj = DigitalTyphoonDataset("/app/datasets/wnp/image/", 
                                    "/app/datasets/wnp/track/", 
                                    "/app/datasets/wnp/metadata.json", 
                                    split_dataset_by='sequence',
                                    load_data_into_memory=False,
                                    ignore_list=[],
                                    verbose=False)

#spliting :
train_data, test_data , validation_data = dataset_obj.random_split([train_size,test_size,validation_size],split_by=split)
print("data size : train = ",len(train_data),", test = ",len(test_data),", validation = ",len(validation_data))

#getting device :
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

#creatting model :
model = model_vgg(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

#training model
training(device,model,loss_fn,optimizer,train_data,test_data,batch_size,epochs)

#testing model
accuracy,cm,f1 = testing(device,model,loss_fn,validation_data,batch_size)


#plotting cm    

df_cm = pd.DataFrame(cm, index=[i for i in range(num_class)],
                        columns=[i for i in range(num_class)])  

curr_time_str = str(datetime.datetime.now().strftime("%d_%m_%Y-%H.%M.%S"))  
plt.figure(figsize=(12, 7))
sn.heatmap(df_cm, annot=True, fmt='g')
plt.savefig(str(path_to_save / 'logs' / f'vgg_confusion_matrix_{curr_time_str}.png'))

#normalize confusion mat
plt.clf()
df_cm = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index=[i for i in range(num_class)],
                        columns=[i for i in range(num_class)])

sn.heatmap(df_cm, annot=True, fmt='g')
plt.savefig(str(path_to_save / 'logs' / f'vgg_confusion_matrix_norm_{curr_time_str}.png'))


#saving model
saving(model,accuracy,epochs,batch_size,cm,f1,path_to_save)
print("Saving done to " + path_to_save)
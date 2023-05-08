import torch
from torch import nn
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

from torch.utils.data import DataLoader

from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset

from model import *
from training import *
from testing import *
from saving import *




#variables :
train_size = 0.8
test_size = 0.1
validation_size = 0.1
num_class = 8
num_workers = 4

if(train_size+test_size+validation_size !=1) : print(f"WARNING : SUM OF SIZES != 1 !!")

split = 'sequence'

batch_size = 32
epochs = 25
curr_time_str = str(datetime.datetime.now().strftime("%d_%m_%Y-%H.%M.%S"))  

path_to_save = f"/app/digtyp/FrameClassification/Vgg/models/model_vgg_sequence_classification_{curr_time_str}_{epochs}_{batch_size}.pth"

print(f'will be saving to :{path_to_save}')
learning_rate= 0.001
loss_fn = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#loading data base
dataset_obj = DigitalTyphoonDataset("/app/datasets/wnp/image/", 
                                    "/app/datasets/wnp/track/", 
                                    "/app/datasets/wnp/metadata.json", 
                                    split_dataset_by='sequence',
                                    load_data_into_memory=False,
                                    labels='grade',
                                    ignore_list=[],
                                    verbose=False)

#spliting :
train_data, test_data , validation_data = dataset_obj.random_split([train_size,test_size,validation_size],split_by=split)
print("data size : train = ",len(train_data),", test = ",len(test_data),", validation = ",len(validation_data))

trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
validationloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#creating model :
model = model_vgg(device,num_class)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#training model
training(device,model,loss_fn,optimizer,trainloader,testloader,batch_size,epochs)

#testing model
accuracy,cm,f1 = testing(device,model,loss_fn,testloader,batch_size)

#accuracy,cm,f1 = testing(device,model,loss_fn,validationloader,batch_size)





print(cm)

#plotting cm    
df_cm = pd.DataFrame(cm, index=[i for i in range(6)],
                        columns=[i for i in range(6)])  

curr_time_str = str(datetime.datetime.now().strftime("%d_%m_%Y-%H.%M.%S"))  
plt.figure(figsize=(12, 7))
sn.heatmap(df_cm, annot=True, fmt='g')
plt.savefig(f'/app/digtyp/FrameClassification/Vgg/models/vgg_seq_confusion_matrix_{curr_time_str}_{epochs}_{batch_size}.png')

#normalize confusion mat
plt.clf()
df_cm = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index=[i for i in range(6)],
                        columns=[i for i in range(6)])

sn.heatmap(df_cm, annot=True, fmt='g')
plt.savefig(f'/app/digtyp/FrameClassification/Vgg/models/vgg_seq_confusion_matrix_norm_{curr_time_str}_{epochs}_{batch_size}.png')


#saving model
saving(model,accuracy,epochs,batch_size,cm,f1,path_to_save)
print("Saving done to " + path_to_save)
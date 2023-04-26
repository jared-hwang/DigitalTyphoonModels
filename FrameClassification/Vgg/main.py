import torch
from torch import nn

from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset

from model import *
from training import *
from testing import *
from saving import *




#variables :
train_size = 0.8
test_size = 0.1
validation_size = 0.1

if(train_size+test_size+validation_size !=1) : print(f"WARNING : SUM OF SIZES != 1 !!")

split = 'frame'

batch_size = 16
epochs = 5
verbose = 2

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
training(device,model,loss_fn,optimizer,train_data,test_data,batch_size,epochs,verbose)

#testing model
accuracy,cm=testing(device,model,loss_fn,validation_data,batch_size,verbose)

#saving model
saving(model,accuracy,epochs,batch_size,cm,path_to_save)

#plotting cm and heatmap
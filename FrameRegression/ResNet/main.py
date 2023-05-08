import torch
from torch import nn
import matplotlib.pyplot as plt
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
num_class = 1
num_workers=0
if(train_size+test_size+validation_size !=1) : print(f"WARNING : SUM OF SIZES != 1 !!")

split = 'frame'
labels_to_train = 'pressure'
batch_size = 32
epochs = 25
path_to_save = f"/app/digtyp/FrameRegression/Vgg/models/model_resnet_regression_{epochs}_{batch_size}.pth"

learning_rate= 0.001
loss_fn = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Warning :  This model has exploding gradient problem")
print(f"Saving to : "+path_to_save)
#loading data base
dataset_obj = DigitalTyphoonDataset("/app/datasets/wnp/image/", 
                                    "/app/datasets/wnp/track/", 
                                    "/app/datasets/wnp/metadata.json", 
                                    split_dataset_by='sequence',
                                    load_data_into_memory=False,
                                    labels=labels_to_train,
                                    ignore_list=[],
                                    verbose=False)

#spliting :
train_data, test_data , validation_data = dataset_obj.random_split([train_size,test_size,validation_size],split_by=split)
print("data size : train = ",len(train_data),", test = ",len(test_data),", validation = ",len(validation_data))


trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
validationloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#creating model :

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#training model
best_mse, best_weight, history = training(device,model,loss_fn,optimizer,trainloader,testloader,batch_size,epochs)

#loading best weights
print("Loading best weights !")
model.load_state_dict(best_weight)


#testing model
mse = testing(device,model,loss_fn,validationloader,batch_size)

#printing results
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
plt.plot(history)
plt.show()

#saving model
saving(model,epochs,batch_size,mse,history,path_to_save)
print("Saving done to " + path_to_save)
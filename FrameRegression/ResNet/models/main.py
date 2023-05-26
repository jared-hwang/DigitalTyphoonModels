import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import datetime
import seaborn as sn


from torch.utils.data import DataLoader
from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset

from training import *
from testing import *
from saving import *

#variables :
train_size = 0.8
test_size = 0.1
validation_size = 0.1
num_class = 1
num_workers=8
if(train_size+test_size+validation_size !=1) : print(f"WARNING : SUM OF SIZES != 1 !!")

split = 'sequence'
labels_to_train = ['grade','wind']
batch_size = 32
epochs = 25


curr_time_str = str(datetime.datetime.now().strftime("%d_%m_%Y_%H")) 
path_to_save = f"/app/digtyp/FrameRegression/ResNet/models/model_resnet_regression_{epochs}_{batch_size}_{curr_time_str}.pth"

learning_rate= 0.001
loss_fn = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Saving to : "+path_to_save)


def image_filter(image):
    return (image.grade() <6 and image.grade()>2)

transform_func = None
def transform_func(image_ray):
    image_ray = np.clip(image_ray, 150, 350)
    image_ray = (image_ray - 150) / (350 - 150)
    image_ray = torch.Tensor(image_ray)
    image_ray = torch.reshape(image_ray, [1, 1, image_ray.size()[0], image_ray.size()[1]])
    image_ray = nn.functional.interpolate(image_ray, size=(224, 224), mode='bilinear', align_corners=False)
    image_ray = torch.reshape(image_ray, [image_ray.size()[2], image_ray.size()[3]])
    image_ray = image_ray.numpy()
    return image_ray

#loading data base
dataset_obj = DigitalTyphoonDataset("/app/datasets/wnp/image/", 
                                    "/app/datasets/wnp/track/", 
                                    "/app/datasets/wnp/metadata.json", 
                                    split_dataset_by='sequence',
                                    load_data_into_memory=False,
                                    labels=labels_to_train,                                    
                                    filter_func=image_filter,
                                    transform_func=transform_func,
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
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,1/1.5)

#training model
best_mse, best_weight, history = training(device,model,loss_fn,optimizer,scheduler,trainloader,testloader,batch_size,epochs)

#loading best weights
print("Loading best weights !")
model.load_state_dict(best_weight)


#testing model
mse, truth,grade, pred = testing(device,model,loss_fn,validationloader,batch_size)

#printing results
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
plt.plot(history)
plt.savefig(f"/app/digtyp/FrameRegression/ResNet/models/model_resnet_regression_{epochs}_{batch_size}_{curr_time_str}.png")
print("fig1 done")

'''
fig, ax = plt.subplots()
x=range(len(truth))
print(len(x),len(truth),len(grade))
sn.scatterplot(x=x, y=truth, hue=grade, ax=ax) 
sn.scatterplot(x=x, y=pred, hue=grade, ax=ax)
fig.savefig(f"/app/digtyp/FrameRegression/ResNet/models/model_resnet_regression_test_{epochs}_{batch_size}_{curr_time_str}.png")    
print("fig2 done")
'''

#saving model
saving(model,epochs,batch_size,mse,history,path_to_save)
print("Saving done to " + path_to_save)
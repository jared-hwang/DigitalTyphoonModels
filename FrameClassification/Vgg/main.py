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
labels_to_train = 'grade'

batch_size = 32
epochs = 25
curr_time_str = str(datetime.datetime.now().strftime("%d_%m_%Y-%H"))  

path_to_save = f"/app/digtyp/FrameClassification/Vgg/models/model_vgg_y_14.18.22_classification_{curr_time_str}_{epochs}_{batch_size}.pth"

print(f'will be saving to :{path_to_save}')
learning_rate= 0.001
loss_fn = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def image_filter(image):
    return (image.grade() <7)

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
'''
train_data, test_data , validation_data = dataset_obj.random_split([train_size,test_size,validation_size],split_by=split)
print("data size : train = ",len(train_data),", test = ",len(test_data),", validation = ",len(validation_data))
'''
#splitting by year
years = dataset_obj.get_years()
nb_years=len(years)
tr_years=[]
te_years=[]
va_years=[]
for i in range(nb_years) :
    if i/nb_years<train_size : 
        tr_years.append(years[i])
    else :
        if i/nb_years<train_size+test_size :
            te_years.append(years[i])
        else :
            va_years.append(years[i])

train_data = dataset_obj.images_from_years(tr_years)
test_data = dataset_obj.images_from_years(te_years)
validation_data = dataset_obj.images_from_years(va_years)
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




print("data size : train = ",len(train_data),", test = ",len(test_data),", validation = ",len(validation_data))
print(f" last year from each list of years : {tr_years[-1]} / {te_years[-1]} / {va_years[-1]}")
print(cm)
n=cm.shape
n=n[0]
#plotting cm    
df_cm = pd.DataFrame(cm, index=[i for i in range(n)],
                        columns=[i for i in range(n)])  

curr_time_str = str(datetime.datetime.now().strftime("%d_%m_%Y-%H"))  
plt.figure(figsize=(12, 7))
sn.heatmap(df_cm, annot=True, fmt='g')
plt.savefig(f'/app/digtyp/FrameClassification/Vgg/models/vgg_y_14.18.22_confusion_matrix_{curr_time_str}_{epochs}_{batch_size}.png')

#normalize confusion mat
plt.clf()
df_cm = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index=[i for i in range(n)],
                        columns=[i for i in range(n)])

sn.heatmap(df_cm, annot=True, fmt='g')
plt.savefig(f'/app/digtyp/FrameClassification/Vgg/models/vgg_y_14.18.22_confusion_matrix_norm_{curr_time_str}_{epochs}_{batch_size}.png')


#saving model
saving(model,accuracy,epochs,batch_size,cm,f1,path_to_save)
print("Saving done to " + path_to_save)
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
#import torchinfo

from torchvision.models import alexnet, vgg16, vgg16_bn
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset
from DigitalTyphoonDataloader.DigitalTyphoonSequence import DigitalTyphoonSequence
from DigitalTyphoonDataloader.DigitalTyphoonImage import DigitalTyphoonImage
from DigitalTyphoonDataloader.DigitalTyphoonUtils import *
from testing import *


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


model = vgg16_bn()

model.features[0]= nn.Conv2d(1,64,kernel_size=(3,3),stride=(1,1),padding=(1,1))
model.features[-1]=nn.AdaptiveMaxPool2d(7*7)
#model.avgpool=nn.AdaptiveAvgPool2d(7*7)
#model.classifier[0]=nn.Linear(in_features = 7*7, out_features=4096, bias = True)
model.classifier[-1]=nn.Linear(in_features = 4096, out_features=10, bias = True)
model=model.to(device)

#torchinfo.summary(model,(3,1,512,512))
#print(model)

dataset_obj = DigitalTyphoonDataset("/app/datasets/wnp/image/", 
                                    "/app/datasets/wnp/track/", 
                                    "/app/datasets/wnp/metadata.json", 
                                    split_dataset_by='sequence',
                                    load_data_into_memory=False,
                                    ignore_list=[],
                                    verbose=False)


ck=torch.load("/app/digtyp/models/firstmodelfullgpu0.pth")
accuracy=ck['accuracy']
#print(f"accuracy= {accuracy}")
model.load_state_dict(ck['model_state_dict'])
loss_fn = nn.CrossEntropyLoss()

train_data, test_data , validation = dataset_obj.random_split([0, 1, 0],split_by='frame')


accuracy,cm=testing(device,model,loss_fn,test_data,16,2)
cm.to("cpu")
torch.save(cm,"/app/digtyp/models/fullcm.pt")
df_cm = pd.DataFrame(cm.numpy(), index = [i for i in "ABCDEFGHIJ"],
                  columns = [i for i in "ABCDEFGHIJ"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

print(f"accuracy = {accuracy}")
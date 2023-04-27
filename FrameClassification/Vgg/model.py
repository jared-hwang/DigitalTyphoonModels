from torch import nn
from torchvision.models import vgg16 , vgg16_bn


def model_vgg(device) :
    model = vgg16_bn()
    model.features[0]= nn.Conv2d(1,64,kernel_size=(3,3),stride=(1,1),padding=(1,1))
    model.features[-1]=nn.AdaptiveMaxPool2d(7*7)
    model.classifier[-1]=nn.Linear(in_features = 4096, out_features=6, bias = True)
    model=model.to(device)
    return model

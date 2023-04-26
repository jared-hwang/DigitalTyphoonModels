import torch
torch.manual_seed(90)
from torch import nn
from tqdm import tqdm

from torchvision.models import vgg16 , vgg16_bn

from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset
from DigitalTyphoonDataloader.DigitalTyphoonSequence import DigitalTyphoonSequence
from DigitalTyphoonDataloader.DigitalTyphoonImage import DigitalTyphoonImage
from DigitalTyphoonDataloader.DigitalTyphoonUtils import *



# See the documentation for description of the optional parameters. 
dataset_obj = DigitalTyphoonDataset("/app/datasets/wnp/image/", 
                                    "/app/datasets/wnp/track/", 
                                    "/app/datasets/wnp/metadata.json", 
                                    split_dataset_by='sequence',
                                    load_data_into_memory=False,
                                    ignore_list=[],
                                    verbose=False)

train_data, test_data , validation = dataset_obj.random_split([0.8, 0.1, 0.1],split_by='frame')
print("data size : train =",len(train_data)," test =",len(test_data))
#print(train_data[1].image().shape)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

epochs = 5
verbose = 2

t=0
accuracy=0

n= len(train_data) #nb of images in training usually n = len(dataloader)
n_test = len(test_data)
batch_size = 32

model = vgg16_bn()
model.features[0]= nn.Conv2d(1,64,kernel_size=(3,3),stride=(1,1),padding=(1,1))
model.features[-1]=nn.AdaptiveMaxPool2d(7*7)
model.classifier[-1]=nn.Linear(in_features = 4096, out_features=10, bias = True)
model=model.to(device)

#print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
#print(device)
print("\n")
while t < epochs and accuracy < 70 : 
    print(f"Epoch {t+1}\n-------------------------------")
    
    #training
    model.train()
    for i in tqdm(range(0,n,batch_size)):
        batch = i/batch_size
        batch_indice = train_data.indices[i:i+batch_size]
        input_image = train_data.dataset.images_as_tensor(batch_indice)
        input_image = input_image.reshape(input_image.shape[0],1,512,512)
        #print(input_image.shape)
        input_image = input_image.to(device)
        output_label = (train_data.dataset.labels_as_tensor(batch_indice,"grade")).to(device)
        
        # Compute prediction error
        pred = model(input_image)
        loss = loss_fn(pred, output_label.long())
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 1000 == 0 and verbose >= 2:
            loss, current = loss.item(), (batch + 1) * len(input_image)
            print(f"loss: {loss:>7f}  [{current:>f}/{n:>5f}]")
    
    print(f"training done !")

    #testing
    model.eval()
    test_loss, correct = 0, 0
    num_batches= n_test /batch_size
    with torch.no_grad():
        for i in range(0,n_test,batch_size):
            batch = i/batch_size            
            batch_indice = test_data.indices[i:i+batch_size]
            input_image = test_data.dataset.images_as_tensor(batch_indice)
            input_image = input_image.reshape(input_image.shape[0],1,512,512).to(device)
            output_label = test_data.dataset.labels_as_tensor(batch_indice,"grade").to(device)
            pred = model(input_image)
            test_loss += loss_fn(pred, output_label.long()).item()
            correct += (pred.argmax(1) == output_label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= n_test
    if verbose >= 1:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    accuracy = 100 * correct
    t+=1
print("Done!")

print("Saving Model :")

torch.save({'model_state_dict':model.state_dict(),
            'accuracy':accuracy,
            'epoch' :epochs,
            'batch_size':batch_size
            },'/app/digtyp/models/firstmodelfullgpu1batch32.pth')

print("Done!")
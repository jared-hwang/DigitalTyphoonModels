from tqdm import tqdm
from testing import *
import numpy as np
import copy

def training(device,model,loss_fn,optimizer,trainloader,testloader,batch_size,epochs):
    t=1
    best_mse = np.inf
    best_weight = None
    history = []
    
    while t <= epochs : 
        
        #training
        model.train()
        with tqdm(trainloader, desc = f'Epoch {t}',dynamic_ncols=True, unit="batch") as pbar:
            for batch_num, data in enumerate(pbar,0):
                
                
                #get image and label
                input_images, input_labels = data
                input_images, input_labels = torch.Tensor(input_images).float(), torch.Tensor(input_labels).float()
                input_images = torch.reshape(input_images, [input_images.size()[0], 1, input_images.size()[1], input_images.size()[2]])
                input_images, input_labels  = input_images.to(device), input_labels.to(device)
                
                # Compute prediction error
                pred = model(input_images)
                loss = loss_fn(pred, input_labels.float())
                #print(f"loss  = {loss}")
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                #print mse in pbar every 100 batch
                #if batch % 100 == 0 :
                pbar.set_postfix({"mse":loss.item()})
                
        
        print(f"    Training done !")

        #testing
        mse = testing(device,model,loss_fn,testloader,batch_size)
        #saving best model
        history.append(mse)
        if mse < best_mse : best_mse, best_weight= mse, copy.deepcopy(model.state_dict())        
        
        t+=1
    print("Done!")
    return best_mse, best_weight ,history

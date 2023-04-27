from tqdm import tqdm
from testing import *

def training(device,model,loss_fn,optimizer,trainloader,testloader,batch_size,epochs) -> float:
    t=1
    accuracy=0
    
    while t <= epochs : 
        
        #training
        model.train()
        with tqdm(trainloader, desc = f'Epoch {t}',dynamic_ncols=True, unit="batch") as pbar:
            for batch_num, data in enumerate(pbar,0):
                
                
                #get image and label
                input_images, input_labels = data
                input_images, input_labels = torch.Tensor(input_images).float(), torch.Tensor(input_labels).long()
                input_images = torch.reshape(input_images, [input_images.size()[0], 1, input_images.size()[1], input_images.size()[2]])
                input_images, input_labels  = input_images.to(device), input_labels.to(device)
                
                # Compute prediction error
                pred = model(input_images)
                loss = loss_fn(pred, input_labels)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                #print loss in pbar every 100 batch
                #if batch % 100 == 0 :
                loss= loss.item()
                pbar.set_postfix({"loss":loss})
                
        
        print(f"    Training done !")

        #testing
        accuracy,cm,f1 = testing(device,model,loss_fn,testloader,batch_size)        
        t+=1
    print("Done!")
    return accuracy ,cm ,f1

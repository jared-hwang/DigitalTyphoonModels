from tqdm import tqdm
from testing import *
import numpy as np
import copy

def training(device,model,loss_fn,optimizer,train_data,validation_data,batch_size,epochs):
    t=1
    n_train = 5000 #len(train_data)
    best_mse = np.inf
    best_weight = None
    history = []
    
    while t <= epochs : 
        
        #training
        model.train()
        with tqdm(range(0,n_train,batch_size), desc = f'Epoch {t}',dynamic_ncols=True, unit="batch") as pbar:
            for i in pbar:
                batch = i/batch_size
                
                #get image
                batch_indice = train_data.indices[i:i+batch_size]
                input_image = train_data.dataset.images_as_tensor(batch_indice)
                input_image = input_image.reshape(input_image.shape[0],1,512,512)
                input_image = input_image.to(device)
                output_label = (train_data.dataset.labels_as_tensor(batch_indice,"pressure")).to(device)
                #print(output_label)
                
                # Compute prediction error
                pred = model(input_image)
                loss = loss_fn(pred, output_label.float())
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
        mse = testing(device,model,loss_fn,validation_data,batch_size)
        
        #saving best model
        history.append(mse)
        if mse < best_mse : best_mse, best_weight= mse, copy.deepcopy(model.state_dict())        
        
        t+=1
    print("Done!")
    return best_mse, best_weight ,history

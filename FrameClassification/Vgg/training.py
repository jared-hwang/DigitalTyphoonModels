from tqdm import tqdm
from testing import *

def training(device,model,loss_fn,optimizer,train_data,validation_data,batch_size,epochs) -> float:
    t=0
    accuracy=0
    n_train = len(train_data)
    
    while t < epochs : 
        
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
                output_label = (train_data.dataset.labels_as_tensor(batch_indice,"grade")).to(device)
                
                # Compute prediction error
                pred = model(input_image)
                loss = loss_fn(pred, output_label.long())
                
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
        accuracy,cm,f1 = testing(device,model,loss_fn,validation_data,batch_size)        
        t+=1
    print("Done!")
    return accuracy ,cm ,f1

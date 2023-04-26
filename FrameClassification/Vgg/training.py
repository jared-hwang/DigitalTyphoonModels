from tqdm import tqdm
from testing import *

def training(device,model,loss_fn,optimizer,train_data,validation_data,batch_size,epochs,verbose) -> float:
    t=0
    accuracy=0
    n_train =len(train_data)
    n_validation = len(validation_data)
    
    while t < epochs : 
        
        #training
        model.train()
        for i in tqdm(range(0,n_train,batch_size), desc = f'Epoch {t}'):
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
            '''
            if batch % 1000 == 0 and verbose >= 2:
                loss, current = loss.item(), (batch + 1) * len(input_image)
                print(f"loss: {loss:>7f}  [{current:>f}/{n_train:>5f}]")
            '''
        
        print(f"training done !")

        #testing
        accuracy,cm = testing(model,loss_fn,validation_data,batch_size,verbose,device)        
        t+=1
    print("Done!")
    return accuracy

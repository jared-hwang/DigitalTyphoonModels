from tqdm import tqdm
import torch

def testing(device,model,loss_fn,test_data,batch_size) :
    #testing
    model.eval()
    mse = 0
    n_test= 1000 #len(test_data)
    num_batches= n_test /batch_size
    
    with torch.no_grad():
        with tqdm(range(0,n_test,batch_size),dynamic_ncols=True,unit="batch",desc="Testing") as pbar:
            for i in pbar:
                #batch = i/batch_size            
                batch_indice = test_data.indices[i:i+batch_size]
                
                #get Image
                input_image = test_data.dataset.images_as_tensor(batch_indice)
                input_image = input_image.reshape(input_image.shape[0],1,512,512).to(device)
                input_label = test_data.dataset.labels_as_tensor(batch_indice,"pressure").to(device)
                
                # Compute prediction error
                pred = model(input_image)
                mse += loss_fn(pred, input_label.long())  
                
                # print accuracy on load bar every 50 batch
                #if batch%50 == 1:
                pbar.set_postfix({'mse':mse})
                    
    mse =mse.float()
    mse /= num_batches
    
    print(f"Test Error: \n Mean Squared Error: {mse:>8f}\n")
    
    
    return  mse
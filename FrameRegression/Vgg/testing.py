from tqdm import tqdm
import torch

def testing(device,model,loss_fn,testloader,batch_size) :
    #testing
    model.eval()
    mse = 0
    num_batches= len(testloader)
    n_test= num_batches * batch_size
    
    with torch.no_grad():
        with tqdm(testloader,dynamic_ncols=True,unit="batch",desc="Testing") as pbar:
            for batch_number,data in enumerate(pbar,0):            
                
                
                #get Image and label
                input_images, input_labels = data
                input_images, input_labels = torch.Tensor(input_images).float(), torch.Tensor(input_labels).long()
                input_images = torch.reshape(input_images, [input_images.size()[0], 1, input_images.size()[1], input_images.size()[2]])
                input_labels = torch.reshape(input_labels, input_labels.size()[0])
                input_images, input_labels  = input_images.to(device), input_labels.to(device)
                
                # Compute prediction error
                pred = model(input_images)
                mse += loss_fn(pred, input_labels.long())  
                
                # print accuracy on load bar every 50 batch
                #if batch%50 == 1:
                pbar.set_postfix({'mse':mse.item()})
                    
    mse = mse.cpu()
    mse = mse.float()
    mse /= num_batches
    
    print(f"Test Error: \n Mean Squared Error: {mse:>8f} per batchs // {mse/batch_size:>8f} average per image\n")
    
    mse = mse/batch_size
    
    return  mse
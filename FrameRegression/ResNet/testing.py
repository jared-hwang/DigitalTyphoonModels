from tqdm import tqdm
import torch

def testing(device,model,loss_fn,testloader,batch_size) :
    #testing
    model.eval()
    mse = 0
    mse_tot = 0
    truth = []
    truth2 = []
    predictions = []
    num_batches = len(testloader)
    n_test = num_batches * batch_size
    
    with torch.no_grad():
        with tqdm(testloader,dynamic_ncols=True,unit="batch",desc="Testing") as pbar:
            for batch_number,data in enumerate(pbar,0):            
                
                
                #get Image and label
                input_images,  input_labels = data
                grade , input_labels = input_labels[:,0], input_labels[:,1]
                input_images, grade ,  input_labels = torch.Tensor(input_images).float(),torch.Tensor(grade).float(), torch.Tensor(input_labels).float()
                input_images = torch.reshape(input_images, [input_images.size()[0], 1, input_images.size()[1], input_images.size()[2]])
                grade = torch.reshape(input_labels, [grade.size()[0],1])
                input_labels = torch.reshape(input_labels, [input_labels.size()[0],1])
                input_images, input_labels  = input_images.to(device), input_labels.to(device)
                
                # Compute prediction error
                pred = model(input_images)
                mse = loss_fn(pred, input_labels.float())  
                mse_tot +=mse
                # print accuracy on load bar every 50 batch
                #if batch%50 == 1:
                pbar.set_postfix({'mse':mse.item()/batch_size})
                
                truth.extend(input_labels.to('cpu'))
                truth2.extend(grade.cpu())
                predictions.extend(pred.to('cpu'))
                
    mse_tot = mse_tot.cpu()
    mse_tot = mse_tot.float()
    mse_tot /= num_batches
    
    print(f"Test Error: \n Mean Squared Error: {mse_tot:>8f} per batchs // {mse_tot/batch_size:>8f} average per image\n")
    
    mse_tot = mse_tot/batch_size
    
    return  mse_tot, truth,truth2, predictions
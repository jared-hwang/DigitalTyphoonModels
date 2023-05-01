from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix,f1_score

def testing(device,model,loss_fn,testloader,batch_size) :
    #testing
    model.eval()
    cm = torch.zeros(10,10,dtype= int).to(device)
    truth_labels = []
    predicted_labels=[]
    test_loss, correct = 0, 0
    num_batches= len(testloader)
    n_test= num_batches * batch_size
    
    with torch.no_grad():
        with tqdm(testloader,dynamic_ncols=True,unit="batch",desc="Testing") as pbar:
            for batch_number,data in enumerate(pbar,0):            
                
                
                #get Image and label
                input_images, input_labels = data
                input_images, input_labels = torch.Tensor(input_images).float(), torch.Tensor(input_labels).long()
                input_images = torch.reshape(input_images, [input_images.size()[0], 1, input_images.size()[1], input_images.size()[2]])
                input_images, input_labels  = input_images.to(device), input_labels.to(device)
                
                # Compute prediction error
                pred = model(input_images)
                test_loss += loss_fn(pred, input_labels).item()
                correct += (pred.argmax(1) == input_labels).type(torch.float).sum().item()
                
                
                
                # print accuracy on load bar every 50 batch
                #if batch%50 == 1:
                accuracy = correct * 100 /((batch_number+1)*batch_size) 
                pbar.set_postfix({'accuracy':accuracy})
                    
                #add labels to a list for confusion matrix later
                predicted_label = torch.argmax(pred, 1).to('cpu')
                predicted_labels.extend(predicted_label)
                truth_labels.extend(input_labels.to('cpu'))
                
    test_loss /= num_batches
    correct /= n_test
    
    #compute confusion matrix
    cm = confusion_matrix(truth_labels,predicted_labels)
    f1 = f1_score(truth_labels,predicted_labels, average="weighted")
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}, F1 Score: {f1} \n")
    accuracy = 100 * correct
    
    
    return accuracy , cm , f1
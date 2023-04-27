from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix,f1_score

def testing(device,model,loss_fn,test_data,batch_size) :
    #testing
    model.eval()
    cm = torch.zeros(10,10,dtype= int).to(device)
    truth_labels = []
    predicted_labels=[]
    test_loss, correct = 0, 0
    n_test= len(test_data)
    num_batches= n_test /batch_size
    
    with torch.no_grad():
        with tqdm(range(0,n_test,batch_size),dynamic_ncols=True,unit="batch",desc="Testing") as pbar:
            for i in pbar:
                batch = i/batch_size            
                batch_indice = test_data.indices[i:i+batch_size]
                
                #get Image
                input_image = test_data.dataset.images_as_tensor(batch_indice)
                input_image = input_image.reshape(input_image.shape[0],1,512,512).to(device)
                input_label = test_data.dataset.labels_as_tensor(batch_indice,"grade").to(device)
                
                # Compute prediction error
                pred = model(input_image)
                test_loss += loss_fn(pred, input_label.long()).item()
                correct += (pred.argmax(1) == input_label).type(torch.float).sum().item()
                
                
                
                # print accuracy on load bar every 50 batch
                #if batch%50 == 1:
                accuracy = correct * 100 /(i+1) 
                pbar.set_postfix({'accuracy':accuracy})
                    
                #add labels to a list for confusion matrix later
                truth_labels.append(float(input_label[0].to('cpu')))
                predicted_label = torch.argmax(pred).to('cpu')
                predicted_labels.append(float(predicted_label))
                
    test_loss /= num_batches
    correct /= n_test
    
    #compute confusion matrix
    cm = confusion_matrix(truth_labels,predicted_labels)
    f1 = f1_score(truth_labels,predicted_labels, average="weighted")
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}, F1 Score: {f1} \n")
    accuracy = 100 * correct
    
    
    return accuracy , cm , f1
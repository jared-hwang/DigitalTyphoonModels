from tqdm import tqdm
import torch
from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset
from DigitalTyphoonDataloader.DigitalTyphoonSequence import DigitalTyphoonSequence
from DigitalTyphoonDataloader.DigitalTyphoonImage import DigitalTyphoonImage
from DigitalTyphoonDataloader.DigitalTyphoonUtils import *

def testing(device,model,loss_fn,test_data,batch_size,verbose) :
    #testing
    model.eval()
    cm = torch.zeros(10,10,dtype= int).to(device)
    test_loss, correct = 0, 0
    n_test=len(test_data)
    num_batches= n_test /batch_size
    with torch.no_grad():
        for i in tqdm(range(0,n_test,batch_size)):
            batch = i/batch_size            
            batch_indice = test_data.indices[i:i+batch_size]
            input_image = test_data.dataset.images_as_tensor(batch_indice)
            input_image = input_image.reshape(input_image.shape[0],1,512,512).to(device)
            input_label = test_data.dataset.labels_as_tensor(batch_indice,"grade").to(device)
            pred = model(input_image)
            test_loss += loss_fn(pred, input_label.long()).item()
            correct += (pred.argmax(1) == input_label).type(torch.float).sum().item()
            for j in range(len(input_label)):
                true_label = input_label[j].int()
                pred_label = pred.argmax(1)[j].int()
                #print(true_label,pred_label)
                cm[true_label, pred_label] += 1
    test_loss /= num_batches
    correct /= n_test
    if verbose >= 1:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    accuracy = 100 * correct
    return accuracy , cm
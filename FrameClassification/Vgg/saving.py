import torch

def saving(model,accuracy,epochs,batch_size,cm,path) :
    print("Saving Model :")

    torch.save({'model_state_dict':model.state_dict(),
                'accuracy':accuracy,
                'epoch' :epochs,
                'batch_size':batch_size,
                'confusion_matrix':cm
                },path)

    print(f"Done !")
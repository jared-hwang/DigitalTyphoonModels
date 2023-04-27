import torch

def saving(model,accuracy,epochs,batch_size,cm,f1,path) :
    print("Saving Model :")

    torch.save({'model_state_dict':model.state_dict(),
                'accuracy':accuracy,
                'epoch' :epochs,
                'batch_size':batch_size,
                'confusion_matrix':cm,
                'f1_score':f1
                },path)

    print(f"Done !")
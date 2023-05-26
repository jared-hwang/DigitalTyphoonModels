import torch

def saving(model,epochs,batch_size,mse,history,path) :
    print("Saving Model :")

    torch.save({'model_state_dict':model.state_dict(),
                'epoch' :epochs,
                'batch_size':batch_size,
                'mse':mse,
                'history':history
                },path)

    print(f"Done !")
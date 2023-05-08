import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import alexnet, vgg16_bn, resnet18, vit_b_16
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn

from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset

def filter(image):
    return (image.grade() < 7)

def set_device(i):
  device = torch.device('cuda:' + str(i) if torch.cuda.is_available() else 'cpu')
  print('device used: ', device)
  return device

def init_alexnet():
  model = alexnet(num_classes=8)
  model.features[0]= nn.Conv2d(1,64,kernel_size=11,stride=4,padding=2)
  return model

def init_vgg16():
    model = vgg16_bn(num_classes=8)
    model.features[0]= nn.Conv2d(1,64,kernel_size=(3,3),stride=(1,1),padding=(1,1))
    model.features[-1]=nn.AdaptiveMaxPool2d(7*7)
    model.classifier[-1]=nn.Linear(in_features = 4096, bias = True)
    return model

def init_resnet():
  model = resnet18(num_classes=7)
  model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  model.fc = nn.Linear(in_features=512, out_features=8, bias=True)
  return model    

def init_vit_net():
  model = vit_b_16(num_classes=8)
  patch_size = 16
  model.conv_proj = nn.Conv2d(in_channels=1, out_channels=768, kernel_size=patch_size, stride=patch_size)
  return model

def cm_compute(confusion_matrix, labels, predicted):
    for j in range(len(labels)):
                  true_label = int(labels[j].item())
                  pred_label = predicted[j].item()
                  confusion_matrix[true_label, pred_label] += 1
    return confusion_matrix

def input_transform(inputs, mean, std, resolution):
    inputs = ((inputs - mean) / std).float()
    assert not torch.any(torch.isnan(inputs))
    inputs = inputs.reshape(inputs.shape[0], 1, 512, 512)
    inputs = nn.functional.interpolate(inputs, size=resolution, mode='bilinear', align_corners=False)
    return inputs

def main():

  SEED = 83
  train_part = 0.8
  device = set_device(1)
  n_epoch = 100
  mean = 269.5767
  std = 34.3959
  image_path = "/home/dataset/image/"
  track_path = "/home/dataset/track/"
  metadata_path = "/home/dataset/metadata.json"
  result_path = "/home/results_vuillod/models_05_03/"


  # Import the database in a dataset_obj
  print('importing dataset...')
  dataset_obj = DigitalTyphoonDataset(image_path,
                                      track_path, 
                                      metadata_path,
                                      'grade',
                                      split_dataset_by='frame',
                                      get_images_by_sequence=False,
                                      load_data_into_memory='all_data',
                                      ignore_list=[],
                                      filter_func=filter,
                                      verbose=False)

  # Split Data
  g1 = torch.Generator().manual_seed(SEED)
  train, test = dataset_obj.random_split([train_part, 1-train_part], split_by='frame', generator=g1)

  train_loader = torch.utils.data.DataLoader(train, batch_size=16,
                                            shuffle=True, num_workers=8)
  test_loader = torch.utils.data.DataLoader(test, batch_size=16,
                                           shuffle=True, num_workers=8)

  # Instantiate the network and define the loss function and optimizer
  model = init_vit_net()
  model_name = 'transformer'
  last_i_saved = 48
  model.load_state_dict(torch.load(result_path + model_name + '/' + 'net0.80_epoch%d.pth'%(last_i_saved) ))

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  
  model = model.to(device)

  # Train the network
  print('%d images loaded in the train_set'% len(train.indices))
  print('Start training' + model_name)
  for epoch in range(last_i_saved +1, n_epoch):
      print('Epoch: ', epoch)
      PATH = result_path + model_name + '/' + 'net%0.2f_epoch%d.pth'% (train_part, epoch)
      print(PATH)
      running_loss = 0.0
      with tqdm(train_loader, dynamic_ncols=True) as pbar:
          for i, data in enumerate(pbar, 0):
              inputs, labels = data
              inputs = torch.clamp(inputs, 150, 350)
              inputs = input_transform(inputs, mean, std, (224, 224))
              inputs = inputs.to(device)
              labels = labels.to(device)

              optimizer.zero_grad()
              outputs = model(inputs)
              loss = criterion(outputs, labels.long())
              loss.backward()
              optimizer.step()
              running_loss += loss.item()

              if loss.isnan():
                  print("Error: exploding gradient issue at image ", i)
                  break

              if i % 2000 == 0:
                  running_loss = 0.0
              else:
                  pbar.set_postfix({'loss': running_loss/(i%2000)})

      if loss.isnan():
          break
      # Save the current model
      torch.save(model.state_dict(), PATH)
      print("model saved with %d epochs and trained with %d%% of the images"% (epoch, train_part*100))


      # TESTING
      print('testing ', PATH)
      correct = 0
      total = 0
      confusion_matrix = torch.zeros(8, 8, dtype=int)

      with tqdm(test_loader, dynamic_ncols=True) as pbar:
          for i, data in enumerate(pbar, 0):
              inputs, labels = data
              inputs = input_transform(inputs, mean, std, (224, 224)).to(device)
              labels = labels.to(device)

              # Prediction
              with torch.no_grad():
                  outputs = model(inputs)
                  _, predicted = torch.max(outputs.data, 1)
                  total += labels.size(0)
                  correct += (predicted == labels).sum().item()

              # Confusion matrix computing
              confusion_matrix = cm_compute(confusion_matrix, labels, predicted)

      # Results
      confusion_matrix = confusion_matrix[2:7, 2:7]
      print('Accuracy of the network with %d epochs on %d%% of test images: %d %%' %
          (epoch, (1-train_part)*100, 100 * correct / total))

      # Save heatmap.png and confusion_matrix tensor
      torch.save(confusion_matrix, result_path + model_name + '/' + 'confusion_matrix_%d.pt'%(epoch))
      sn.heatmap(confusion_matrix, annot=True, fmt='g', xticklabels=range(2,7), yticklabels=range(2,7))
      plt.savefig(result_path + model_name + '/' + f'confusion_matrix_{epoch}.png')
      plt.close()

  print('Finished Training')

if __name__ == "__main__":
    main()
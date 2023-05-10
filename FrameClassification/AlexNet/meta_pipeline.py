from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset
import torch

from classification import train_and_test

def set_device(i):
  device = torch.device('cuda:' + str(i) if torch.cuda.is_available() else 'cpu')
  return device

def data_filter(image):
    return (image.grade() < 7)

def main():
    SEED = 83
    train_part = 0.8
    device = set_device(1)
    n_epochs = 100
    mean = 269.5767
    std = 34.3959
    learning_rate = 0.0001
    momentum = 0.9
    result_path = "/home/results_vuillod/models_05_09/"
    model_name='resnet18'
    weights='DEFAULT'

    # Import the database in a dataset_obj
    print('importing dataset...')
    image_path = "/home/dataset/image/"
    track_path = "/home/dataset/track/"
    metadata_path = "/home/dataset/metadata.json"

    dataset_obj = DigitalTyphoonDataset(image_path,
                                        track_path, 
                                        metadata_path,
                                        'grade',
                                        split_dataset_by='sequence',
                                        get_images_by_sequence=False,
                                        load_data_into_memory='all_data',
                                        ignore_list=[],
                                        filter_func=data_filter,
                                        verbose=False)
    
    train_and_test(dataset_obj, device, 
                   model_name, weights,
                   SEED, train_part,
                   n_epochs, 
                   mean, std, 
                   learning_rate, momentum,
                   result_path
                   )

if __name__ == "__main__":
    main()
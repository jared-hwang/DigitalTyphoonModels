import pytorch_lightning as pl
import numpy as np
import torch 
import math
from tqdm import tqdm
from torch import nn
from hyperparameters import *
from PadLabels import PadSequence
from PositionalEncoding import PositionalEncoding
from SequenceDatamodule import TyphoonDataModule
from lightning_transformer_pressure_forecast import LightningTransformerLabelsOnly
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
plt.style.use('seaborn-whitegrid')

def remove_pad_from_tensor(full_labels, pad_token):
    # Remove padding
    first_pad = 0
    for i in range(len(full_labels)):
        if full_labels[i] == pad_token:
            first_pad = i
            break
    full_truth = full_labels[:first_pad]
    return full_truth

def predict(model, full_labels, pred_start_pct, device, SOS_token, EOS_token, pad_token=4):
    # Produce predictions

    # Remove padding
    full_truth = remove_pad_from_tensor(full_labels, pad_token)

    # Make src 
    forecast_start_idx = int((len(full_truth)-1) * pred_start_pct) + 1
    end_idx = len(full_truth)

    src = full_truth[:forecast_start_idx].to(device)
    
    # Prediction loop
    # tgt = src[-1:].long().to(device)
    tgt = src[1:forecast_start_idx].long().to(device)

    start_idx = forecast_start_idx

    for _ in range(start_idx, end_idx):
        # Get source mask
        tgt_mask = model.get_tgt_mask(tgt.size()[0])
        
        pred = model(torch.reshape(src, (1, src.size()[0])), torch.reshape(tgt, (1, tgt.size()[0])), tgt_mask)
        
        next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
        next_item = torch.tensor([next_item], device=device)

        # Concatenate previous input with predicted best word
        tgt = torch.cat((tgt, next_item))

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS_token:
            break

    return tgt.view(-1).tolist()

def produce_prediction(model, batch, pred_start_pct, device, SOS_token, EOS_token, pad_token):
    sequence, src, tgt, tgt_expected = batch
    
    truths = []
    predictions = []
    forecast_start_idx = []
    # Iterate for each sample in the batch
    for i in range(len(src)):
        truths.append(remove_pad_from_tensor(src[i], pad_token)[1:]) # [1:] to remove sos_token
        predictions.append(predict(model, src[i], pred_start_pct, device, SOS_token, EOS_token, pad_token))
        forecast_start_idx.append(int(len(truths[-1])*pred_start_pct))

    return truths, predictions, forecast_start_idx

def plot_predictions(truths, predictions, forecast_start_idx, save_path):
    forecast_end_idx = len(truths)
    fig = plt.figure()
    ax = plt.axes()

    x = list(range(len(truths)))
    ax.plot(x, truths, label='Ground truth')
    plot_end_idx = forecast_end_idx
    predictions_to_plot = predictions[forecast_start_idx-1:plot_end_idx] # -1 to show the last item that should match with truth
    predictions_x = x[forecast_start_idx-1:plot_end_idx]    
    ax.plot(predictions_x, predictions_to_plot, label="Forecast")
    
    ax.set_title('Pressure by hour', fontweight ="bold")
    ax.set_ylabel('Pressure ')
    ax.set_xlabel('Hour')
    ax.legend()
    fig.savefig(save_path)
    plt.close()

def read_validation_indices(filepath):
    with open(filepath, 'r') as f:
        line = f.readlines()[0][1:-1]
        indices_list = [int(num) for num in line.split(',')]
    return indices_list

def evaluate_saved_model(checkpoint_path, validation_indices, log_path, plot_every=10):
    model = LightningTransformerLabelsOnly.load_from_checkpoint(checkpoint_path)

    # disable randomness, dropout, etc...
    model.eval()

    data_module = TyphoonDataModule(data_dir,
                                batch_size=plot_every,
                                num_workers=num_workers,
                                labels='pressure',
                                split_by=split_by,
                                load_data=load_data,
                                dataset_split=dataset_split,
                                standardize_range=standardize_range,
                                transform=transforms.Compose([
                                            PadSequence(max_sequence_length, 
                                                        PAD_token, SOS_token, EOS_token,
                                                        label_range=(min_pressure, max_pressure),
                                                        img_range=img_range),
                                ]),
                                downsample_size=downsample_size)
    data_module.setup(0)
    val_set = Subset(data_module.dataset, validation_indices)
    val_loader = DataLoader(val_set, batch_size=plot_every, num_workers=num_workers, shuffle=False)
    for i, batch in enumerate(tqdm(val_loader)):
        truths, predictions, forecast_start_indices = produce_prediction(model, batch, 0.75, model.device, SOS_token, EOS_token, pad_token=PAD_token)
    
        # Only plot first in batch
        plot_predictions(truths[0], predictions[0], forecast_start_indices[0], log_path + f'test_plot_{i}.png')

checkpoint_path = '/DigitalTyphoonModels/FrameClassification/ResNet/lightning_logs/label_only_forecast_logs/lightning_logs/version_11/checkpoints/epoch=59-step=10560.ckpt'
validation_path = '/DigitalTyphoonModels/FrameClassification/ResNet/lightning_logs/label_only_forecast_logs/lightning_logs/version_11/validation_indices.txt'
logging_path = '/DigitalTyphoonModels/FrameClassification/ResNet/lightning_logs/label_only_forecast_logs/lightning_logs/version_11/plots/'
#logging_path = ''

from lightning_transformer_pressure_forecast_crossentropy import LightningTransformerLabelsOnly

num_workers=0
img_range = (0, 0)

validation_indices = read_validation_indices(validation_path)
evaluate_saved_model(checkpoint_path, validation_indices, logging_path, plot_every=3)

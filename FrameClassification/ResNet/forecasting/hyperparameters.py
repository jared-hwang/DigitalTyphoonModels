import numpy as np
import torch

# Hyperparameters
learning_rate     = 0.1
batch_size        = 5
num_workers       = 16
max_epochs        = 100
weights           = None
split_by          = 'sequence'
load_data         = False
dataset_split     = (0.8, 0.2, 0.0)
standardize_range = (170, 300)
downsample_size   = (128, 128)
num_heads         = 24
num_encoder_layers = 8
num_decoder_layers = 8
dropout_p         = 0.1
SOS_token         = np.array([2])
EOS_token         = np.array([3])
PAD_token         = 4
accelerator       = 'gpu' if torch.cuda.is_available() else 'cpu'

data_dir          = '/data/'
log_dir           = "/DigitalTyphoonModels/FrameClassification/ResNet/lightning_logs/label_only_forecast_logs/"

max_pressure = 1020  # truly 1018
min_pressure = 860   # truly 870
# min_temp, max_temp = 170, 300
min_temp, max_temp = 0, 0

num_tokens = 10*(max_pressure - min_pressure) + 5  # +5 to account for eos, sos, and pad
dim_model = 528
max_sequence_length = 528

prediction_start_point_pct = 0.75

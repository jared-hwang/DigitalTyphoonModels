import torch

# Training Hyperparameters
LEARNING_RATE     = 0.0001
BATCH_SIZE        = 16
NUM_WORKERS       = 16
MAX_EPOCHS        = 30

# DATASET
WEIGHTS           = None
SPLIT_BY          = 'sequence'
LOAD_DATA         = 'all_data'
DATASET_SPLIT     = (0.8, 0.1, 0.1)
STANDARDIZE_RANGE = (150, 350)
DOWNSAMPLE_SIZE   = (224, 224)
NUM_CLASSES       = 5

# Computation
ACCELERATOR       = 'gpu' if torch.cuda.is_available() else 'cpu'
DATA_DIR          = '/dataset/'
LOG_DIR           = "/DigitalTyphoonModels/FrameClassification/Alexnet/lightning/tb_logs"
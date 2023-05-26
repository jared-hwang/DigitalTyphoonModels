import torch

# Training Hyperparameters
LEARNING_RATE     = 0.0001
BATCH_SIZE        = 16
NUM_WORKERS       = 16
MAX_EPOCHS        = 76


# DATASET
WEIGHTS           = None
LABELS            = 'wind'
SPLIT_BY          = 'sequence'
LOAD_DATA         = 'all_data'
DATASET_SPLIT     = (0.8, 0.1, 0.1)
STANDARDIZE_RANGE = (170, 350)
DOWNSAMPLE_SIZE   = (224, 224)
NUM_CLASSES       = 1

# Computation
ACCELERATOR       = 'gpu' if torch.cuda.is_available() else 'cpu'
DATA_DIR          = '/app/datasets/wnp/'
LOG_DIR           = "/app/digtyp/FrameClassification/Vgg/lightning/tb_logs"
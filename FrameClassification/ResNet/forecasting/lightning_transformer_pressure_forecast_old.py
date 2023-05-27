import os
import matplotlib.pyplot as plt
import seaborn as sn
import io
import numpy as np
import pandas as pd
from PIL import Image
import datetime
import math
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
from hyperparameters import *


from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset
from SequenceDatamodule import TyphoonDataModule
from PositionalEncoding import PositionalEncoding
from logging_utils import Logger

class LightningTransformerLabelsOnly(pl.LightningModule):
    def __init__(self, num_tokens, 
                       dim_model,
                       num_heads,
                       num_encoder_layers, 
                       num_decoder_layers,
                       dropout_p, 
                       learning_rate,
                       max_sequence_length,
                       label_range=(860, 1020),
                       temp_range=(170,300),
                       SOS_token=np.array([2]),
                       EOS_token=np.array([3]),
                       PAD_token=4):
        super().__init__()
        self.save_hyperparameters()

        # Hyperparams
        self.learning_rate = learning_rate
        self.dim_model = dim_model

        # Define Model
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=48
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model = dim_model, 
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p
        )
        self.out = nn.Linear(dim_model, num_tokens)
        # self.out = nn.Linear(dim_model, 1)

        # Loss functions and statistics
        #self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)
        self.criterion = nn.MSELoss()

        # Misc data vars
        self.max_sequence_length = max_sequence_length
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.PAD_token = PAD_token
        self.label_range = label_range
        self.temp_range = temp_range


        self.total_val_loss = 0

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)        
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # we permute to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
        # out = transformer_out
        return out

    def training_step(self, train_batch, batch_idx):
        loss = self._common_step(train_batch, batch_idx)
        self.log('train_loss', loss) #, on_step=False, on_epoch=True)
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        loss = self._common_step(val_batch, batch_idx)
        self.log('val_loss', loss) #, on_step=False, on_epoch=True)
        return loss
        
    def _common_step(self, batch, batch_idx):
        sequence, src, tgt, tgt_expected = batch
        batch_size, tgt_length = tgt.size()[0], tgt.size()[1]

        # Generate masking matrixes
        tgt_mask = self.get_tgt_mask(tgt_length)
        tgt_pad_mask = self.create_pad_mask(tgt, self.PAD_token)
        src_pad_mask = self.create_pad_mask(src, self.PAD_token)
        
        pred = self.forward(src, tgt, tgt_mask=tgt_mask, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)
        loss = self.loss_fn(pred, tgt_expected)
        self.total_val_loss += loss.item()
        return loss

    def on_validation_epoch_end(self):
        # collect stats
        self.log('total_validation_loss', self.total_val_loss)
        self.total_val_loss = 0

        if self.current_epoch == 0:
            version = self.logger.version
            val_set = self.trainer.datamodule.val_set
            with open(str(Path(log_dir) / 'lightning_logs' / f'version_{version}' / f'validation_indices.txt'), 'w') as f:
                f.write(str([int(idx) for idx in val_set.indices]))

        return
     
    def loss_fn(self, predictions, labels):
        loss_mask = (labels != self.PAD_token)
        # print(predictions.size(), labels.size())
        # predictions = (loss_mask * torch.argmax(predictions, dim=1)).float() # extract most likely token prediction
        predictions = torch.reshape(predictions, labels.size())
        predictions = (predictions*loss_mask).float()
        labels = (labels*loss_mask).float()
        # print(predictions)
        return self.criterion(predictions, labels)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a square matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1).to(self.device) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

    def map_label_to_rep(self, labels):
        labels = labels - self.label_range[0] + 4 + self.temp_range[1]
        return labels
    
    def map_img_to_rep(self, img):
        img = img - self.temp_range[0] + 4
        return img

    def map_label_rep_to_label(self, label_rep):
        labels = label_rep - 4 + self.label_range[0] + self.temp_range[1]
        return labels
    
    def map_img_rep_to_label(self, img_rep):
        img = img - 4 + self.temp_range[0]
        return img



# JSONlogger.write_json(str(Path(log_dir) / 'lightning_logs' / f'version_{version}' / f'labels_only_forecast_json_{start_time_str}.json'))
# https://towardsdatascience.com/how-to-run-inference-with-a-pytorch-time-series-transformer-394fd6cbe16c
# https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1


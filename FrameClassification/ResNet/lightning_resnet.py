import numpy as np
import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchmetrics import F1Score, ConfusionMatrix, Accuracy
import os
import matplotlib.pyplot as plt
import seaborn as sn
import io
from PIL import Image
import datetime
from pathlib import Path
from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset
from FrameDatamodule import TyphoonDataModule
from logging_utils import Logger

JSONlogger = Logger()
start_time_str = str(datetime.datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))


class LightningResnet(pl.LightningModule):

    def __init__(self, learning_rate, weights, num_classes):
        super().__init__()
        self.save_hyperparameters()

        # Hyperparams
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Define Model
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=weights)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(in_features=512, out_features=7, bias=True)

        # Loss functions and statistics
        self.criterion = nn.CrossEntropyLoss()
        self.compute_micro_f1 = F1Score(task="multiclass", num_classes=num_classes, average='micro')
        self.compute_macro_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.compute_weighted_f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.compute_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)

        # Collected statistics
        self.predicted_labels = []
        self.truth_labels = []

        # Log hyperparams



    def forward(self, images):
        # Add channel
        images = torch.Tensor(images).float()
        images = torch.reshape(images, [images.size()[0], 1, images.size()[1], images.size()[2]])
        output = self.resnet(images)
        return output

    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        labels = labels - 2
        predictions = self.forward(images)
        loss = self.cross_entropy_loss(predictions, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch
        labels = labels - 2
        predictions = self.forward(images)
        loss = self.cross_entropy_loss(predictions, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        self.predicted_labels.append(torch.argmax(predictions, 1))
        self.truth_labels.append(labels.int())

        return loss

    def on_validation_epoch_end(self):
        all_preds = torch.concat(self.predicted_labels)
        all_truths = torch.concat(self.truth_labels)

        micro_f1_result = self.compute_micro_f1(all_preds, all_truths)
        macro_f1_result = self.compute_macro_f1(all_preds, all_truths)
        weighted_f1_result = self.compute_weighted_f1(all_preds, all_truths)
        accuracy = self.accuracy(all_preds, all_truths)

        self.log('micro_f1', micro_f1_result)
        self.log('macro_f1', macro_f1_result)
        self.log('weighted_f1', weighted_f1_result)
        self.log('accuracy', accuracy)
        cm = self.log_confusion_matrix(all_preds, all_truths)

        JSONlogger.log_json_and_txt_pairs(self.current_epoch,
                                          [('val_acc', float(accuracy)),
                                           ('val_microf1', float(micro_f1_result)),
                                           ('val_macrof1', float(macro_f1_result)),
                                           ('val_weightedf1', float(weighted_f1_result))])
        JSONlogger.log_json(self.current_epoch, 'confusion_matrix', cm.tolist())

        self.predicted_labels.clear()  # free memory
        self.truth_labels.clear()
    
    def cross_entropy_loss(self, predictions, labels):
        labels = torch.Tensor(labels).long()
        return self.criterion(predictions, labels)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        return optimizer

    def log_confusion_matrix(self, all_preds, all_truths):
        # https://stackoverflow.com/questions/65498782/how-to-dump-confusion-matrix-using-tensorboard-logger-in-pytorch-lightning/73388839#73388839
        tb = self.logger.experiment

        cf_matrix = self.compute_cm(all_preds, all_truths)
        computed_confusion = cf_matrix.detach().cpu().numpy().astype(int)

        df_cm = pd.DataFrame(computed_confusion, index=[i + 2 for i in range(self.num_classes)],
                         columns=[i + 2 for i in range(self.num_classes)])


        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=.65)
        ax.set_title(f'Epoch: {self.current_epoch}'); 
        sn.set(font_scale=1.2)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)
        buf = io.BytesIO()

        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        im = Image.open(buf)
        im = transforms.ToTensor()(im)
        tb.add_image("val_confusion_matrix", im, global_step=self.current_epoch)
        plt.close()
        return cf_matrix


# Hyperparameters
learning_rate     = 0.0001
batch_size        = 16
num_workers       = 16
max_epochs        = 1
weights           = None
split_by          = 'sequence'
load_data         = False
dataset_split     = (0.8, 0.2, 0.0)
dataset_split     = (0.1, 0.1, 0.8)
standardize_range = (150, 350)
downsample_size   = (224, 224)
num_classes       = 5
accelerator       = 'gpu' if torch.cuda.is_available() else 'cpu'
data_dir          = '/data/'
log_dir           = "/DigitalTyphoonModels/FrameClassification/ResNet/lightning_logs/sequence_logs/"

# Set up data
data_module = TyphoonDataModule(data_dir,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                split_by=split_by,
                                load_data=load_data,
                                dataset_split=dataset_split,
                                standardize_range=standardize_range,
                                downsample_size=downsample_size)


# Train
model = LightningResnet(learning_rate=learning_rate, weights=weights, num_classes=num_classes)
trainer = pl.Trainer(accelerator=accelerator, max_epochs=max_epochs, default_root_dir=log_dir)

trainer.fit(model, data_module)


# Log things to JSON
version = trainer.logger.version
JSONlogger.log_json_and_txt_pairs('meta', [('start_time', start_time_str),
                                    ('num_max_epochs', max_epochs),
                                    ('batch_size', batch_size),
                                    ('lr', learning_rate),
                                    ('split_by', split_by),
                                    ('standardized', standardize_range),
                                    ('downsample', downsample_size),
                                    ('weights', weights)])

JSONlogger.write_json(str(Path(log_dir) / 'lightning_logs' / f'version_{version}' / f'resnet_json_{start_time_str}.json'))

import datetime
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from lightning_transformer_pressure_forecast import LightningTransformerLabelsOnly
from hyperparameters import *
from PadLabels import PadSequence
from SequenceDatamodule import TyphoonDataModule
from logging_utils import Logger

JSONlogger = Logger()
start_time_str = str(datetime.datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))

data_module = TyphoonDataModule(data_dir,
                                batch_size=batch_size,
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
                                                        img_range=standardize_range,
                                ]),
                                downsample_size=downsample_size)

# Train
model = LightningTransformerLabelsOnly(num_tokens, 
                                       dim_model,
                                       num_heads,
                                       num_encoder_layers, 
                                       num_decoder_layers,
                                       dropout_p, 
                                       learning_rate,
                                       max_sequence_length=max_sequence_length,
                                       SOS_token=SOS_token,
                                       EOS_token=EOS_token,
                                       PAD_token=PAD_token)

checkpoint_callback = ModelCheckpoint(monitor='total_validation_loss', mode='min', every_n_epochs=1, save_top_k=5)
trainer = pl.Trainer(accelerator=accelerator, max_epochs=max_epochs, default_root_dir=log_dir, callbacks=[checkpoint_callback])

trainer.fit(model, data_module)

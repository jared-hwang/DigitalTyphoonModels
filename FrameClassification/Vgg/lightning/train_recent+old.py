import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from FrameDatamodule import TyphoonDataModule
from lightning_vgg import LightningVgg
import config
import loading
import torch
from torch import nn
from torch.utils.data import DataLoader

from pathlib import Path
import numpy as np

from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset

from datetime import datetime

start_time_str = str(datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))

def main():
    logger = TensorBoardLogger("tb_logs", name="vgg_double")

    # Set up data
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    standardize_range=config.STANDARDIZE_RANGE,
    downsample_size=config.DOWNSAMPLE_SIZE,

    batch_size=batch_size[0]
    num_workers=num_workers[0]
    standardize_range=standardize_range[0]
    downsample_size=downsample_size[0]

    data_path = Path("/app/datasets/wnp/")
    images_path = str(data_path / "image") + "/"
    track_path = str(data_path / "track") + "/"
    metadata_path = str(data_path / "metadata.json")

    def image_filter(image):
        return (
            (image.grade() < 7)
            and (image.year() != 2023)
            and (100.0 <= image.long() <= 180.0)
        )  # and (image.mask_1_percent() <  self.corruption_ceiling_pct))

    def transform_func(image_ray):
        image_ray = np.clip(
            image_ray,standardize_range[0],standardize_range[1]
        )
        image_ray = (image_ray - standardize_range[0]) / (
            standardize_range[1] - standardize_range[0]
        )
        if downsample_size != (512, 512):
            image_ray = torch.Tensor(image_ray)
            image_ray = torch.reshape(
                image_ray, [1, 1, image_ray.size()[0], image_ray.size()[1]]
            )
            image_ray = nn.functional.interpolate(
                image_ray,
                size=downsample_size,
                mode="bilinear",
                align_corners=False,
            )
            image_ray = torch.reshape(
                image_ray, [image_ray.size()[2], image_ray.size()[3]]
            )
            image_ray = image_ray.numpy()
        return image_ray

    dataset = DigitalTyphoonDataset(
                str(images_path),
                str(track_path),
                str(metadata_path),
                "grade",
                load_data_into_memory='all_data',
                filter_func=image_filter,
                transform_func=transform_func,
                spectrum="Infrared",
                verbose=False,
            )



    train_old,test_old = loading.load(0,dataset,batch_size,num_workers)
    train_recent,test_recent = loading.load(1,dataset,batch_size,num_workers)
    train_now,test_now = loading.load(2,dataset,batch_size,num_workers)    
    train_double,test_double = loading.load(3,dataset,batch_size,num_workers)
    
    # Train
    model = LightningVgg(
        learning_rate=config.LEARNING_RATE,
        weights=config.WEIGHTS,
        num_classes=config.NUM_CLASSES,
    )
    
    trainer = pl.Trainer(
        logger=logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICE,
        max_epochs=config.MAX_EPOCHS,
        default_root_dir=config.LOG_DIR,
    )
    '''
    print("Training <2005")
    trainer_old.fit(model_old, train_old, test_old)
    
    
    print("Training >2005")
    trainer_recent.fit(model_recent, train_recent, test_recent)
    '''
    print("Training >2015")    
    trainer.fit(model, train_now, test_now)
    
    print("Test on :")
    print("<2005")
    trainer.test(model,dataloaders=test_old)
    


if __name__ == "__main__":
    main()

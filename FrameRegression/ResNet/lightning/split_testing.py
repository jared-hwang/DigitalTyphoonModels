import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from FrameDatamodule import TyphoonDataModule
from lightning_resnetReg import LightningResnetReg
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
    logger_old = TensorBoardLogger("tb_logs", name="resnet_test_old_same")
    logger_recent = TensorBoardLogger("tb_logs", name="resnet_test_recent_same")
    logger_now = TensorBoardLogger("tb_logs", name="resnet_test_now_same")

    # Set up data
    batch_size=config.BATCH_SIZE
    num_workers=config.NUM_WORKERS
    standardize_range=config.STANDARDIZE_RANGE
    downsample_size=config.DOWNSAMPLE_SIZE    
    type_save = config.TYPE_SAVE

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
                "pressure",
                load_data_into_memory='all_data',
                filter_func=image_filter,
                transform_func=transform_func,
                spectrum="Infrared",
                verbose=False,
            )


    _,test_old = loading.load(0,dataset,batch_size,num_workers,type_save)
    _,test_recent = loading.load(1,dataset,batch_size,num_workers,type_save)
    _,test_now = loading.load(2,dataset,batch_size,num_workers,type_save)
    
    # Test
    
    trainer_old = pl.Trainer(
        logger=logger_old,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICE,
        max_epochs=config.MAX_EPOCHS,
        default_root_dir=config.LOG_DIR,
    )
    
    trainer_recent = pl.Trainer(
        logger=logger_recent,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICE,
        max_epochs=config.MAX_EPOCHS,
        default_root_dir=config.LOG_DIR,
    )

    trainer_now = pl.Trainer(
        logger=logger_now,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICE,
        max_epochs=config.MAX_EPOCHS,
        default_root_dir=config.LOG_DIR,
    )
    
    i=0

    model_old = LightningResnetReg.load_from_checkpoint(f"tb_logs/resnet_train_old_same/version_{i}/checkpoints/epoch=100-step=167155.ckpt")
    model_recent = LightningResnetReg.load_from_checkpoint(f"tb_logs/resnet_train_recent_same/version_{i}/checkpoints/epoch=100-step=196546.ckpt")
    model_now = LightningResnetReg.load_from_checkpoint(f"tb_logs/resnet_train_now_same/version_{i}/checkpoints/epoch=100-step=203515.ckpt")
    
    print("Testing <2005")
    print("         on <2005 : ")
    trainer_old.test(model_old, test_old)
    print("         on >2005 : ")
    trainer_old.test(model_old, test_recent)
    print("         on >2015 : ")
    trainer_old.test(model_old, test_now)
    
    print("Testing >2005")
    print("         on <2005 : ")
    trainer_recent.test(model_recent, test_old)    
    print("         on >2005 : ")
    trainer_recent.test(model_recent, test_recent)
    print("         on >2015 : ")
    trainer_recent.test(model_recent, test_now)
    
    print("Testing >2015")
    print("         on <2005 : ")
    trainer_now.test(model_now, test_old)
    print("         on >2005 : ")
    trainer_now.test(model_now, test_recent)
    print("         on >2015 : ")
    trainer_now.test(model_now, test_now)
    print(f"Run {i} done")
    

if __name__ == "__main__":
    main()

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from FrameDatamodule import TyphoonDataModule
from lightning_resnet import LightningResnet
import config

from datetime import datetime

start_time_str = str(datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))

def main():
    logger = TensorBoardLogger("tb_logs", name="resnet_v0")

    # Set up data
    data_module = TyphoonDataModule(
        config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        split_by=config.SPLIT_BY,
        load_data=config.LOAD_DATA,
        dataset_split=config.DATASET_SPLIT,
        standardize_range=config.STANDARDIZE_RANGE,
        downsample_size=config.DOWNSAMPLE_SIZE,
    )

    # Train
    model = LightningResnet(
        learning_rate=config.LEARNING_RATE,
        weights=config.WEIGHTS,
        num_classes=config.NUM_CLASSES,
    )
    trainer = pl.Trainer(
        logger=logger,
        accelerator=config.ACCELERATOR,
        devices=[1],
        max_epochs=config.MAX_EPOCHS,
        default_root_dir=config.LOG_DIR,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()

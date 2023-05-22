import torch.nn as nn
import torch
import torch.optim as optim
from torchvision.models import resnet18
import pytorch_lightning as pl
from torchmetrics import F1Score, ConfusionMatrix, Accuracy
import torchvision


class LightningResnet(pl.LightningModule):
    def __init__(self, learning_rate, weights, num_classes):
        super().__init__()
        self.save_hyperparameters()

        self.model = resnet18(num_classes=7, weights=weights)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.model.fc = nn.Linear(in_features=512, out_features=8, bias=True)

        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()

        self.compute_micro_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="micro"
        )
        self.compute_macro_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.compute_weighted_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="weighted"
        )
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.compute_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, images):
        images = torch.Tensor(images).float()
        images = torch.reshape(
            images, [images.size()[0], 1, images.size()[1], images.size()[2]]
        )
        output = self.model(images)
        return output

    def training_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch)
        accuracy = self.accuracy(outputs, labels)
        self.log_dict({
            "train_loss": loss, 
            "train_accuracy": accuracy
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # if batch_idx % 10000 == 0:
        #     x , y = batch
        #     grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
        #     self.logger.experiment.add_image("typhoon_images", grid, self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch)
        self.log("validation_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch)
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def _common_step(self, batch):
        images, labels = batch
        labels = labels - 2
        outputs = self.forward(images)
        loss = self.loss_fn(outputs, labels.long())
        outputs = torch.argmax(outputs, 1)
        return loss, outputs, labels

    def predict_step(self, batch):
        images, labels = batch
        labels = labels - 2
        outputs = self.forward(images)
        preds = torch.argmax(outputs, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.learning_rate)

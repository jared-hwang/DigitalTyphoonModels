import torch.nn as nn
import torch
import torch.optim as optim
from torchvision.models import resnet18
import pytorch_lightning as pl
from torchmetrics import F1Score, ConfusionMatrix, Accuracy, MeanSquaredError
import torchvision


class LightningResnetReg(pl.LightningModule):
    def __init__(self, learning_rate, weights, num_classes):
        super().__init__()
        self.save_hyperparameters()

        self.model = resnet18(num_classes=1, weights=weights)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
        
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()
        self.accuracy = MeanSquaredError(squared = False)
        
        self.predicted_labels = []
        self.truth_labels = []


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
            "train_RMSE": accuracy
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
        self.log("validation_loss", loss,
            on_step=False, on_epoch=True, sync_dist=True)
        self.predicted_labels.append(outputs)
        self.truth_labels.append(labels.float())
        return loss

    def test_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch)
        self.log("test_loss", loss,
            on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def _common_step(self, batch):
        images, labels = batch
        labels = labels - 2
        labels = torch.reshape(labels, [labels.size()[0],1])
        outputs = self.forward(images)
        loss = self.loss_fn(outputs, labels.float())
        return loss, outputs, labels

    def predict_step(self, batch):
        images, labels = batch
        labels = labels - 2
        labels = torch.reshape(labels, [labels.size()[0],1])
        outputs = self.forward(images)
        preds = outputs
        return preds

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.learning_rate)
    
    def on_validation_epoch_end(self):
        all_preds = torch.concat(self.predicted_labels)
        all_truths = torch.concat(self.truth_labels)
        accuracy = self.accuracy(all_preds, all_truths)
        self.log('validation_RMSE', accuracy,
            on_step=False, on_epoch=True, sync_dist=True)
        
        #print regression line graph every 5 epochs
        if(self.current_epoch %5 == 0 ):
            tensorboard = self.logger.experiment
            
            #print(self.predicted_labels)
            for i in range(len(self.predicted_labels)):
                for j in range(len(self.predicted_labels[i])):
                    tensorboard.add_scalars(f"epoch_{self.current_epoch}",{'pred':self.predicted_labels[i][j],'truth':self.truth_labels[i][j]},self.truth_labels[i][j])
        
        self.predicted_labels.clear()  # free memory
        self.truth_labels.clear()
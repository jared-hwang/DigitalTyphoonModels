import torch.nn as nn
import torch
import torch.optim as optim
from torchvision.models import vgg16_bn
import pytorch_lightning as pl
from torchmetrics import F1Score, ConfusionMatrix, Accuracy
import torchvision
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import io
from PIL import Image
from torchvision import transforms


class LightningVgg(pl.LightningModule):
    def __init__(self, learning_rate, weights, num_classes):
        super().__init__()
        self.save_hyperparameters()

        self.model = vgg16_bn(num_classes=7, weights=weights)
        self.model.features[0]= nn.Conv2d(1,64,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.model.features[-1]=nn.AdaptiveMaxPool2d(7*7)
        self.model.classifier[-1]=nn.Linear(in_features = 4096, out_features=num_classes, bias = True)
        
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = num_classes

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
        
        self.predicted_labels = []
        self.truth_labels = []
        self.compt = 1


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

        self.predicted_labels.append(outputs)
        self.truth_labels.append(labels.int())
        return loss

    def test_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch)
        self.log(f"test_loss_{self.compt}", loss, on_epoch=True, sync_dist=True)

        self.predicted_labels.append(outputs)
        self.truth_labels.append(labels.int())
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
        self.log_confusion_matrix(all_preds, all_truths)
        
        self.predicted_labels.clear()  # free memory
        self.truth_labels.clear()
        
    def on_test_epoch_end(self):
        all_preds = torch.concat(self.predicted_labels)
        all_truths = torch.concat(self.truth_labels)

        micro_f1_result = self.compute_micro_f1(all_preds, all_truths)
        macro_f1_result = self.compute_macro_f1(all_preds, all_truths)
        weighted_f1_result = self.compute_weighted_f1(all_preds, all_truths)
        accuracy = self.accuracy(all_preds, all_truths)
        name=f'test_confusion_matrix_{self.compt}'
    
        self.compt+=1
        self.log('micro_f1', micro_f1_result)
        self.log('macro_f1', macro_f1_result)
        self.log('weighted_f1', weighted_f1_result)
        self.log('accuracy', accuracy)
        self.log_confusion_matrix(all_preds, all_truths, name)
        
        self.predicted_labels.clear()  # free memory
        self.truth_labels.clear()
        
    def log_confusion_matrix(self, all_preds, all_truths, name="val_confusion_matrix"):
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
        tb.add_image(name , im, global_step=self.current_epoch)
        plt.close()

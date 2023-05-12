import torch
import torch.nn as nn
import torchvision.models as models

class TemporalResNet(nn.Module):
    def __init__(self, num_classes, input_sequence_length):
        super(TemporalResNet, self).__init__()

        # Load pre-trained ResNet-18 model
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Remove the average pooling and fully connected layers of the original ResNet-18
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Add temporal pooling layer
        self.temporal_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.input_sequence_length = input_sequence_length

        # Add linear layer for classification
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size, sequence_length, _, _, _ = x.size()

        # Reshape input to (batch_size * sequence_length, channels, height, width)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))

        # Pass input through the ResNet-18 backbone
        features = self.resnet(x)

        # Apply temporal pooling
        features = self.temporal_pooling(features)
        features = features.view(batch_size, sequence_length, -1)

        # Pass features through linear classifier
        logits = self.classifier(features)

        return logits

# # Instantiate the model
# num_classes = 10  # Number of output classes
# input_sequence_length = 5  # Length of input sequence
# model = TemporalResNet(num_classes, input_sequence_length)

# # Generate a random input sequence tensor for testing
# batch_size = 2
# sequence_length = input_sequence_length
# height, width = 224, 224
# channels = 3
# x = torch.randn(batch_size, sequence_length, channels, height, width)

# # Forward pass
# logits = model(x)

# # Print the output shape
# print(logits.shape)

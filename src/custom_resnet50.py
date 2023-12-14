import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet


class CustomResNet50(ResNet):
    def __init__(self):
        super().__init__(Bottleneck, [3, 4, 6, 3])

        # Replace the last layer
        self.fc = nn.Linear(2048, 1)

        # Replace the activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = super().forward(x)
        x = self.sigmoid(x)
        return x

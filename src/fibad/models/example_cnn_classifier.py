# ruff: noqa: D101, D102

# This example model is taken from the PyTorch CIFAR10 tutorial:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa N812
import torch.optim as optim

# extra long import here to address a circular import issue
from fibad.models.model_registry import fibad_model


@fibad_model
class ExampleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.confv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # ~ The following methods are placeholders for future work
    # ~ I don't think this will be the final API!!!
    def criterion(self):
        return nn.CrossEntropyLoss()

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def save(self, path):
        torch.save(self.state_dict(), path)

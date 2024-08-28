# ruff: noqa: D101, D102

# This example model is taken from the PyTorch CIFAR10 tutorial:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa N812
import torch.optim as optim

from .model_registry import fibad_model

logger = logging.getLogger(__name__)


@fibad_model
class ExampleCNN(nn.Module):
    def __init__(self, model_config, shape):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.config = model_config

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self, trainloader, device=None):
        self.optimizer = self._optimizer()
        self.criterion = self._criterion()

        for epoch in range(self.config.get("epochs", 2)):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:
                    logger.info(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}")
                    running_loss = 0.0

    # Creating a train_step function to be used with pytorch-ignite
    # ! figure out how to pass `device` correctly!!! It shouldn't be in the method signature
    # ! I just put it there to pass linting.
    def train_step(self, batch, device):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

    def _criterion(self):
        return nn.CrossEntropyLoss()

    def _optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def save(self):
        torch.save(self.state_dict(), self.config.get("weights_filepath", "example_cnn.pth"))
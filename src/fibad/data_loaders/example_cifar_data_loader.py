# ruff: noqa: D101, D102

import torch
import torchvision
import torchvision.transforms as transforms

from .data_loader_registry import fibad_data_loader


@fibad_data_loader
class CifarDataLoader:
    def __init__(self, data_loader_config):
        self.config = data_loader_config

    def shape(self):
        return (3, 32, 32)

    def get_data_loader(self):
        """This is the primary method for this class.

        Returns
        -------
        torch.utils.data.DataLoader
            The dataloader to use for training.
        """
        return self.data_loader(self.data_set())

    def data_set(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        return torchvision.datasets.CIFAR10(
            root=self.config.get("path", "./data"), train=True, download=True, transform=transform
        )

    def data_loader(self, data_set):
        return torch.utils.data.DataLoader(
            data_set,
            batch_size=self.config.get("batch_size", 4),
            shuffle=self.config.get("shuffle", True),
            num_workers=self.config.get("num_workers", 2),
        )

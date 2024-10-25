# ruff: noqa: D101, D102
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from .data_set_registry import fibad_data_set


@fibad_data_set
class CifarDataSet(CIFAR10):
    """This is simply a version of CIFAR10 that has our needed shape method, and is initialized using
    FIBAD config with a transformation that works well for example code.
    """

    def __init__(self, config, split: str):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        if split not in ["train", "test"]:
            RuntimeError("CIFAR10 dataset only supports 'train' and 'test' splits.")

        train = split == "train"

        super().__init__(root=config["general"]["data_dir"], train=train, download=True, transform=transform)

    def shape(self):
        return (3, 32, 32)

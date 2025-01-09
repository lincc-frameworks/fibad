# ruff: noqa: D101, D102
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
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

        if split not in ["train", "validate", "test"]:
            RuntimeError("CIFAR10 dataset only supports 'train' and 'test' splits.")

        train = split != "test"

        super().__init__(root=config["general"]["data_dir"], train=train, download=True, transform=transform)

        if train:
            num_train = len(self)
            indices = list(range(num_train))
            split_idx = 0
            if config["data_set"]["validate_size"]:
                split_idx = int(np.floor(config["data_set"]["validate_size"] * num_train))

            random_seed = None
            if config["data_set"]["seed"]:
                random_seed = config["data_set"]["seed"]
            np.random.seed(random_seed)
            np.random.shuffle(indices)

            train_idx, valid_idx = indices[split_idx:], indices[:split_idx]

            #! These two "samplers" are used by PyTorch's DataLoader to split the
            #! dataset into training and validation sets. Using Samplers is mutually
            #! exclusive with using "shuffle" in the DataLoader.
            #! If a user doesn't define a Sampler, the default behavior of pytorch-ignite
            #! is to shuffle the data unless `shuffle = False` in the config.
            self.train_sampler = SubsetRandomSampler(train_idx)
            self.validation_sampler = SubsetRandomSampler(valid_idx)

    def shape(self):
        return (3, 32, 32)

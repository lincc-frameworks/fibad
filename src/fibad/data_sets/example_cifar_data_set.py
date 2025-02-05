# ruff: noqa: D101, D102
import logging

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import CIFAR10

from .data_set_registry import fibad_data_set

logger = logging.getLogger(__name__)


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
            self.train_sampler = None
            self.validation_sampler = None
            data_set_length = len(self)
            indices = list(range(data_set_length))

            train_size = config["data_set"]["train_size"] if config["data_set"]["train_size"] else None
            validate_size = (
                config["data_set"]["validate_size"] if config["data_set"]["validate_size"] else None
            )

            if isinstance(train_size, int):
                train_size = train_size / data_set_length
            if isinstance(validate_size, int):
                validate_size = validate_size / data_set_length

            # Shuffle the indices
            random_seed = None
            if config["data_set"]["seed"]:
                random_seed = config["data_set"]["seed"]
            np.random.seed(random_seed)
            np.random.shuffle(indices)

            # Get absolute numbers of training and validation samples
            num_train = 0
            if train_size:
                num_train = int(np.floor(train_size * data_set_length))

            num_validation = 0
            if validate_size:
                num_validation = int(np.floor(validate_size * data_set_length))

            # Let the user know if they are asking for too many points
            if num_train + num_validation > data_set_length:
                raise RuntimeError(
                    f"Sum of train and validation ({num_train} + {num_validation})"
                    f" exceed dataset size ({data_set_length})."
                )

            # Create the SubsetRandomSamplers
            train_idx = []
            if train_size:
                train_idx = indices[:num_train]
                self.train_sampler = SubsetRandomSampler(train_idx)

            valid_idx = []
            if validate_size:
                valid_idx = indices[num_train : num_train + num_validation]
                self.validation_sampler = SubsetRandomSampler(valid_idx)

            logger.debug(f"Dataset size: {len(self)}")
            logger.debug(f"Train size: {len(train_idx)}")
            logger.debug(f"Validation size: {len(valid_idx)}")

    def shape(self):
        return (3, 32, 32)

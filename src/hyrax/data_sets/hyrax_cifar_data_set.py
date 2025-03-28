# ruff: noqa: D101, D102
import logging

import numpy as np
import torchvision.transforms as transforms
from astropy.table import Table
from torchvision.datasets import CIFAR10

from hyrax.config_utils import ConfigDict

from .data_set_registry import HyraxDataset

logger = logging.getLogger(__name__)


class HyraxCifarDataSet(HyraxDataset, CIFAR10):
    """This is simply a version of CIFAR10 that has our needed shape method, and is initialized using
    Hyrax config with a transformation that works well for example code.

    We only use the training split in the data, because it is larger (50k images). Hyrax will then divide that
    into Train/test/Validate according to configuration.
    """

    def __init__(self, config: ConfigDict):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        CIFAR10.__init__(
            self, root=config["general"]["data_dir"], train=True, download=True, transform=transform
        )
        metadata_table = Table({"label": np.array([self[index][1] for index in range(len(self))])})
        super().__init__(config, metadata_table)

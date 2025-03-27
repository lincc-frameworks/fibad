# ruff: noqa: D101, D102
import logging
from collections.abc import Generator
from typing import Optional

import numpy as np
import numpy.typing as npt
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from hyrax.config_utils import ConfigDict

from .data_set_registry import Dataset

logger = logging.getLogger(__name__)


class HyraxCifarDataSet(Dataset, CIFAR10):
    """This is simply a version of CIFAR10 that has our needed shape method, and is initialized using
    Hyrax config with a transformation that works well for example code.

    We only use the training split in the data, because it is larger (50k images). Hyrax will then divide that
    into Train/test/Validate according to configuration.
    """

    def __init__(self, config: ConfigDict):
        super().__init__(config)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        CIFAR10.__init__(
            self, root=config["general"]["data_dir"], train=True, download=True, transform=transform
        )

    def ids(self) -> Generator[str]:
        return (str(x) for x in range(len(self)))

    def shape(self):
        return (3, 32, 32)

    def original_config(self) -> ConfigDict:
        return self.config

    def metadata_fields(self) -> list[str]:
        return ["label"]

    def metadata(self, idxs: npt.ArrayLike, fields: Optional[list[str]] = None) -> npt.ArrayLike:
        if np.isscalar(idxs):
            idxs = np.array([idxs])

        if fields is None:
            fields = self.metadata_fields()

        if len(fields) == 0 or fields != ["label"]:
            msg = "For CifarDataSet 'label' is the only supported field."
            raise RuntimeError(msg)

        return np.array([self[index][1] for index in np.array(idxs)])

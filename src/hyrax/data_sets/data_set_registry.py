# ruff: noqa: D102, B027
import logging
from collections.abc import Generator
from typing import Optional

import numpy.typing as npt
from astropy.table import Table

from hyrax.config_utils import ConfigDict
from hyrax.plugin_utils import get_or_load_class, update_registry

logger = logging.getLogger(__name__)
DATA_SET_REGISTRY: dict[str, type["HyraxDataset"]] = {}


class HyraxDataset:
    """
    How to make a hyrax dataset:

    Start from this template and implement your functions.

    ```
    from torch.utils.data import Dataset
    class MyDataset(HyraxDataset, Dataset):
        def __init__(self, config: dict):
            super().__init__(config)

        def __getitem__():
            # Your getitem goes here
            pass

        def __len__ ():
            # Your len function goes here
            pass

    ```

    <guidance on what each function should do>

    # Overriding IDs

    < How to implement the ids() function as a generator of strings>

    # Metadata Interface (simple)

    < How to pass us a numpy rec array (or astropy table) >

    # Metadata interface (complex)

    <How to implement metadata() and metadata_fields()>

    # Data members provided by the superclass

    `self.config` This is the hyrax config nested dictionary that hyrax was launched with.

    """

    def __init__(self, config: ConfigDict, metadata_table: Optional[Table] = None):
        """Overall initialization for all DataSets which saves the config

        Subclasses of HyraxDataSet ought call this at the end of their __init__ like:

        ```
        from hyrax.data_sets import HyraxDataset
        from torch.utils.data import Dataset

        class MyDataset(HyraxDataset, Dataset):
            def __init__(config):
                <your code>
                super().__init__(config)
        ```

        If per tensor metadata is available, it is recommended that dataset authors create an
        astropy Table of that data, in the same order as their data and pass that `metadata_table`
        as shown below:

        ```
        from hyrax.data_sets import HyraxDataset
        from torch.utils.data import Dataset
        from astropy.table import Table

        class MyDataset(HyraxDataset, Dataset):
            def __init__(config):
                <your code>
                metadata_table = Table(<Your catalog data goes here>)
                super().__init__(config, metadata_table)
        ```

        Parameters
        ----------
        config : ConfigDict, Optional
            The runtime configuration for hyrax
        metadata_table : Optional[Table], optional
            An Astropy Table with
            1. the metadata columns desired for visualization AND
            2. in the order your data will be enumerated.
        """
        self._config = config
        self._metadata_table = metadata_table
        self.tensorboardx_logger = None

    @property
    def config(self):
        return self._config

    def __init_subclass__(cls):
        from torch.utils.data import IterableDataset

        if IterableDataset in cls.__bases__ or hasattr(cls, "__iter__"):
            logger.error("Hyrax does not fully support iterable data sets yet. Proceed at your own risk.")

        # Paranoia. Deriving from a torch dataset class should ensure this, but if an external dataset author
        # Forgets to to do that, we tell them.
        if (not hasattr(cls, "__iter__")) and not (hasattr(cls, "__getitem__") and hasattr(cls, "__len__")):
            msg = f"Hyrax data set {cls.__name__} is missing required iteration functions."
            msg += "__len__ and __getitem__ (or __iter__) must be defined. It is recommended to derive from"
            msg += " torch.utils.data.Dataset (or torch.utils.data.IterableDataset) which will enforce this."
            raise RuntimeError(msg)

        # TODO?:If the subclass has __iter__ and not __getitem__/__len__ perhaps add an __iter__ with a
        #       warning Because to the extent the __getitem__/__len__ functions get used they'll exhaust the
        #       iterator and possibly remove any benefit of having them around.

        # TODO?:If the subclass has __getitem__/__len__ and not __iter__ add an __iter__. This is less
        #       dangerous, and should probably just be an info log.
        #
        #       This might be better as a function on this base class, but doing it here gives us an
        #       opportunity to do configuration or logging to help people navigate writing a dataset?

        # Ensure the class is in the registry so the config system can find it
        update_registry(DATA_SET_REGISTRY, cls.__name__, cls)

    def ids(self) -> Generator[str]:
        """This is the default IDs function you get when you derive from hyrax Dataset

        Returns
        -------
        Generator[str]
            A generator yielding all the string IDs of the dataset.

        """
        if hasattr(self, "__len__"):
            for x in range(len(self)):
                yield str(x)
        elif hasattr(self, "__iter__"):
            for index, _ in enumerate(iter(self)):
                yield (str(index))
        else:
            return NotImplementedError("You must define __len__ or __iter__ to use automatic id()")

    def shape(self) -> tuple:
        """Returns the shape tuple of the tensors this dataset will return.

        This default implementation uses the first item in the dataset to determine the shape.

        Returns
        -------
        tuple
            Shape tuple of the tensor that will be returned from the dataset.
        """
        if hasattr(self, "__getitem__"):
            return self[0].shape
        elif hasattr(self, "__iter__"):
            return next(iter(self)).shape
        else:
            return NotImplementedError("You must define __getitem__ or __iter__ to use automatic shape()")

    def metadata_fields(self) -> list[str]:
        """Returns a list of metadata fields supported by this object

        Returns
        -------
        list[str]
            The column names of the metadata table passed. Empty string if no metadata was provided at
            during construction of the HyraxDataset (or derived class).
        """
        return [] if self._metadata_table is None else list(self._metadata_table.colnames)

    def metadata(self, idxs: npt.ArrayLike, fields: list[str]) -> npt.ArrayLike:
        """Returns a table representing the metadata given an array of indexes and a list of fields.

        Parameters
        ----------
        idxs : npt.ArrayLike
            The indexes of the relevant tensor objects
        fields : list[str]
            The names of the fields you would like returned. All values must be among those returned by
            metadata_fields()

        Returns
        -------
        npt.ArrayLike
            A numpy record array of your metadata, with only the columns specified.
            Roughly equivalent to: `metadata_table[idxs][fields].as_array()` where metadata_table is the
            astropy table that the HyraxDataset (or derived class) was constructed with.

        Raises
        ------
        RuntimeError
            When none of the provided fields are
        """
        metadata_fields = self.metadata_fields()
        for field in fields:
            if field not in metadata_fields:
                msg = f"Field {field} is not available for {self.__class__.__name__}."
                logger.error(msg)

        columns = [field for field in fields if field in metadata_fields]

        if len(columns) == 0:
            msg = (
                f"None of the metadata fields passed [{fields}] are available for {self.__class__.__name__}."
            )
            raise RuntimeError(msg)

        return self._metadata_table[idxs][columns].as_array()


def fetch_data_set_class(runtime_config: dict) -> type[HyraxDataset]:
    """Fetch the data loader class from the registry.

    Parameters
    ----------
    runtime_config : dict
        The runtime configuration dictionary.

    Returns
    -------
    type
        The data loader class.

    Raises
    ------
    ValueError
        If a built in data loader was requested, but not found in the registry.
    ValueError
        If no data loader was specified in the runtime configuration.
    """

    data_set_config = runtime_config["data_set"]
    data_set_cls = None

    try:
        data_set_cls = get_or_load_class(data_set_config, DATA_SET_REGISTRY)
    except ValueError as exc:
        raise ValueError("Error fetching data set class") from exc

    return data_set_cls

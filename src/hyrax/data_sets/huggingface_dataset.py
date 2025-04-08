import abc
from collections.abc import Generator
from typing import Optional

from datasets import Dataset as HFDataset
from datasets import IterableDataset as HFIterableDataset
from torch import Tensor

from .data_set_registry import HyraxDataset


class HyraxHFDatasetBase(HyraxDataset, abc.ABC):
    """
    Base class for multimodal universe datasets.

    Unifies common config, tensor lookup, and id generation functionality. These are essentially the user
    facing pieces of the iterable, map, and streaming versions of these classes.
    """

    def __init__(self, config: dict):
        super().__init__(self, config)
        print("HyraxHFDatasetBase __init__")

        # Parse config: All of these are required
        # TODO: error checking here
        self.hf_config = config["data_set.HuggingFace"]
        dataset_name = self.hf_config["dataset"]
        max_size = self.hf_config["max_size"]
        self.dataset_dict_keys = list(self.hf_config["dict_keys"])

        # Set up the dataset, calling subclass if necessary
        hf_dataset = self._create_dataset(dataset_name, split="train")
        hf_dataset = hf_dataset.with_format("torch")
        if max_size:
            hf_dataset = hf_dataset.take(max_size)

        # Allow the subclass to do whatever initialization is necessary
        # depending on what HF class it is inherited from
        self._become_hf(hf_dataset.select_columns(self.dataset_dict_keys[0]))

        # Create our IDs
        self.id_dataset = hf_dataset.select_columns("object_id")

    def _create_dataset(self, dataset_name, **kwargs):
        from datasets import load_dataset

        return load_dataset(dataset_name, **kwargs)

    @abc.abstractmethod
    def _become_hf(self):
        raise NotImplementedError(
            "Subclasses of MMUDatasetBase must call __init__ of the appropriate HF class themselves"
        )

    def ids(self) -> Generator[str]:
        """
        Generator of ids for all HyraxHFDataSets

        Yields
        ------
        Generator[str]
            Yields object ID strings
        """
        iterator = iter(self.id_dataset)
        return (item["object_id"] for item in iterator)

    def _lookup_tensor(self, dset_object: dict, dict_keys: Optional[list[str]] = None) -> Tensor:
        # TODO: This method of lookup is not very flexible.
        #
        # multimodal dataset HSC cutouts have a dict structure that looks like:
        # {
        #  image: {
        #     band: <list of HSC filters>
        #     flux: <tensor>
        #     ivar:
        #     psf_fwhm:
        #     mask:
        #     scale:
        #  }
        #  object_id:
        #  ...
        # }
        #
        # In general HuggingFace prefers this structure, at least enforcing it at the top
        # level of the dict.
        #
        # One could imagine:
        # 1) wanting all of this metadata as input which would require some overhaul
        # of our contract between models and the rest of fibad. e.g. the invariant of
        # "model inputs are always a tensor with a given shape" would no longer be true.
        #
        # 2) wanting more than one of the tensors packed a particular way. This would
        # break the configuration I've set in this version of the code
        #
        # 3) wanting only a single layer of the tensor. This is currently
        # disallowed, because dataset_dict_keys is a list of strings, but could be enabled
        # with some clever stuffing of integer indexes into strings at the config level
        #
        dict_keys = self.dataset_dict_keys if dict_keys is None else dict_keys

        current = dset_object
        for key in dict_keys:
            current = current[key]
        return current


class HyraxHFMapDataSet(HyraxHFDatasetBase, HFDataset):
    """
    Map style HuggingFace Dataset
    """

    def _become_hf(self, hf_dataset):
        HFDataset.__init__(
            self,
            arrow_table=hf_dataset._data,
            info=hf_dataset.info,
            split=hf_dataset.split,
            indices_table=hf_dataset._indices,
        )
        self.set_format("torch")

    def __getitem__(self, idxs):
        """
        Get an item from the dataset, can handle a list of indexes

        Parameters
        ----------
        idxs : Union[int, list[int]]
            index or list of indexes

        Returns
        -------
        torch.Tensor
            The data itself
        """
        # First lookup required by HF
        item = super().__getitem__(idxs)[self.dataset_dict_keys[0]]
        items = [item] if not isinstance(item, list) else item
        tensors = [self._lookup_tensor(item, self.dataset_dict_keys[1:]) for item in items]
        retval = tensors if len(tensors) > 1 else tensors[0]
        return retval

    def __getitems__(self, idxs):
        # HF (and pytorch) use the not-python-standard
        # __getitems__ as an interface to get batches.
        return self.__getitem__(idxs)


class HyraxHFIterableDataSet(HyraxHFDatasetBase, HFIterableDataset):
    """
    Iterable style HuggingFace Dataset
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.batch_size = config["data_loader"]["batch_size"]

    def _become_hf(self, hf_dataset):
        HFIterableDataset.__init__(
            self,
            ex_iterable=hf_dataset._ex_iterable,
            info=hf_dataset.info,
            split=hf_dataset.split,
            formatting=hf_dataset._formatting,
            shuffling=hf_dataset._shuffling,
            distributed=hf_dataset._distributed,
            token_per_repo_id=hf_dataset._token_per_repo_id,
        )

    def _create_dataset(self, dataset_name, **kwargs):
        from datasets import load_dataset

        hf_dataset = load_dataset(dataset_name, split="train")
        return hf_dataset.to_iterable_dataset()

    # TODO: the major issue with this code is that iterating is non-performant
    # Below are two implementations of __iter__(), Each of them take about a second
    # to evaluate the superclass iterator we depend on for data
    #
    #  The performance in a notebook is ~5ms for the same iterator, and
    #  I suspect something about the interaction between this code and the torch
    #  dataloader may be to blame; however, there is no root cause at time of writing.

    # non-batch implementation
    # def __iter__(self) -> Generator[Tensor]:
    #     import time
    #     iterator = super().__iter__()
    #     start_time_ns = time.monotonic_ns()
    #     for item in iterator:
    #         time_ms = (time.monotonic_ns() - start_time_ns)/1_000_000
    #         print(f"Iterator took {time_ms} ms")
    #         tensor = self._lookup_tensor(item)
    #         time_ms = (time.monotonic_ns() - start_time_ns)/1_000_000
    #         print(f"Lookup took {time_ms} ms")
    #         yield tensor
    #         print("Begin clock")
    #         start_time_ns = time.monotonic_ns()

    # batch implementation
    def __iter__(self):
        for batch in super().iter(batch_size=self.batch_size):
            batch = batch[self.dataset_dict_keys[0]]
            tensor_batch = [self._lookup_tensor(item, self.dataset_dict_keys[1:]) for item in batch]
            print(type(tensor_batch[0]))
            print(tensor_batch[0].shape)
            for tensor in tensor_batch:
                yield tensor

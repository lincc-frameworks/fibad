import logging
from collections.abc import Generator
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Union

import numpy as np
from torch import Tensor, from_numpy
from torch.utils.data import Dataset

from fibad.config_utils import find_most_recent_results_dir

from .data_set_registry import fibad_data_set

logger = logging.getLogger(__name__)


@fibad_data_set
class InferenceDataSet(Dataset):
    """This is a dataset class to represent the situations where we wish to treat the output of inference
    as a dataset. e.g. when performing umap/visualization operations"""

    def __init__(
        self,
        config,
        results_dir: Optional[Union[Path, str]] = None,
        verb: Optional[str] = None,
    ):
        self.config = config
        self.results_dir = self._resolve_results_dir(config, results_dir, verb)

        # Open the batch index numpy file.
        # Loop over files and create if it does not exist
        batch_index_path = self.results_dir / "batch_index.npy"
        if not batch_index_path.exists():
            msg = f"{self.results_dir} is corrupt and lacks a batch index file."
            raise RuntimeError(msg)

        self.batch_index = np.load(self.results_dir / "batch_index.npy")
        self.length = len(self.batch_index)

        # Initializes our first element. This primes the cache for sequential access
        # as well as giving us a sample element for shape()
        self.cached_batch_num: Optional[int] = None
        self.shape_element = self._load_from_batch_file(
            self.batch_index["batch_num"][0], self.batch_index["id"][0]
        )[0]

    def shape(self):
        """The shape of the dataset (Discovered from files)

        Returns
        -------
        Tuple
            Tuple with the shape of an individual element of the dataset
        """
        return self.shape_element["tensor"].shape

    def ids(self) -> Generator[str]:
        """IDs of this dataset

        Returns
        -------
        Generator[str]
            Generator that yields the string ids of this dataset

        Yields
        ------
        Generator[str]
            Yields the string ids of this dataset
        """
        return (str(id) for id in self.batch_index["id"])

    def __getitem__(self, idx: Union[int, np.ndarray]) -> Tensor:
        try:
            _ = (e for e in idx)  # type: ignore[union-attr]
        except TypeError:
            idx = np.array([idx])

        # Allocate a numpy array to hold all the tensors we will get in order
        # Needs to be the appropriate shape
        shape_tuple = tuple([len(idx)] + list(self.shape()))
        all_tensors = np.zeros(shape=shape_tuple)

        # We need to look up all the batches for the ids we get
        lookup_batch = self.batch_index[idx]

        # We then need to sort the resultant id->batch catalog by batch
        original_indexes = np.argsort(lookup_batch, order="batch_num")
        sorted_lookup_batches = np.take_along_axis(lookup_batch, original_indexes, axis=-1)

        unique_batch_nums = np.unique(sorted_lookup_batches["batch_num"])
        for batch_num in unique_batch_nums:
            # Mask our batch out to get IDs and the original indexes it had in the query
            batch_mask = sorted_lookup_batches["batch_num"] == batch_num
            batch_ids = sorted_lookup_batches[batch_mask]["id"]
            batch_original_indexes = original_indexes[batch_mask]

            # Lookup in each batch file
            batch_tensors = self._load_from_batch_file(batch_num, batch_ids)

            # Place the resulting tensors in the results array where they go.
            all_tensors[batch_original_indexes] = batch_tensors["tensor"]

        # In the case of a single id this will be a tensor that has the appropriate shape
        # Otherwise we will have a stacked array of tensors
        all_tensors = all_tensors[0] if len(all_tensors) == 1 else all_tensors

        return from_numpy(all_tensors)

    def __len__(self) -> int:
        return self.length

    def _load_from_batch_file(self, batch_num: int, ids=Union[int, np.ndarray]) -> np.ndarray:
        """Hands back an array of tensors given a set of IDs in a particular batch and the given
        batch number"""

        # Ensure the cached batch is loaded
        if self.cached_batch_num is None or batch_num != self.cached_batch_num:
            self.cached_batch_num = batch_num
            self.cached_batch: np.ndarray = np.load(self.results_dir / f"batch_{batch_num}.npy")

        return self.cached_batch[np.isin(self.cached_batch["id"], ids)]

    def _resolve_results_dir(
        self, config, results_dir: Optional[Union[Path, str]], verb: Optional[str]
    ) -> Path:
        """Initialize an inference results directory as a data source. Accepts an override of what
        directory to use"""

        verb = "infer" if verb is None else verb

        if results_dir is None:
            if self.config["results"]["inference_dir"]:
                results_dir = self.config["results"]["inference_dir"]
                if not isinstance(results_dir, str):
                    msg = "Configured [results_dir] is not a string"
                    raise RuntimeError(msg)
            else:
                results_dir = find_most_recent_results_dir(self.config, verb=verb)
                if results_dir is None:
                    msg = "Could not find a results directory. Run infer or use "
                    msg += "[results] inference_dir config to specify a directory."
                    raise RuntimeError(msg)
                msg = f"Using most recent results dir {results_dir} for lookup."
                msg += " Use the [results] inference_dir config to set a directory or pass it to this verb."
                logger.info(msg)

        retval = Path(results_dir) if isinstance(results_dir, str) else results_dir

        if not retval.exists():
            msg = f"Inference directory {results_dir} does not exist"
            raise RuntimeError(msg)

        return retval


class InferenceDataSetWriter:
    """Class to write out inference datasets. Used by infer, umap to consistently write out numpy
    files in batches which can be read by InferenceDataSet.

    With the exception of building ID->Batch indexing info, this is implemented as a bag-o-functions that
    manipulate the filesystem directly as their primary effect.
    """

    def __init__(self, result_dir: Union[str, Path]):
        self.result_dir = result_dir if isinstance(result_dir, Path) else Path(result_dir)
        self.batch_index = 0

        self.all_ids = np.array([], dtype=np.int64)
        self.all_batch_nums = np.array([], dtype=np.int64)
        self.writer_pool = Pool()

    def write_batch(self, ids: np.ndarray, tensors: list[np.ndarray]):
        """Write a batch of tensors into the dataset. This writes the whole batch immediately.
        Caller is in charge of batch size consistency considerations, and that ids is the same length as
        tensors

        Parameters
        ----------
        ids : np.ndarray
            Array of integer IDs.
        tensors : list[np.ndarray]
            List of consistently dimensioned numpy arrays to save.
        """
        batch_len = len(tensors)

        # Save results from this batch in a numpy file as a structured array
        first_tensor = tensors[0]
        structured_batch_type = np.dtype(
            [("id", np.int64), ("tensor", first_tensor.dtype, first_tensor.shape)]
        )
        structured_batch = np.zeros(batch_len, structured_batch_type)
        structured_batch["id"] = ids
        structured_batch["tensor"] = tensors

        filename = f"batch_{self.batch_index}.npy"
        savepath = self.result_dir / filename
        if savepath.exists():
            RuntimeError(f"Writing objects in batch {self.batch_index} but {filename} already exists.")

        self.writer_pool.apply_async(
            func=np.save, args=(savepath, structured_batch), kwds={"allow_pickle": False}
        )

        self.all_ids = np.append(self.all_ids, ids)
        self.all_batch_nums = np.append(self.all_batch_nums, np.full(batch_len, self.batch_index))

        self.batch_index += 1

    def write_index(self):
        """Writes out the batch index built up by this object over multiple write_batch calls.
        See save_batch_index for details.
        """
        # First ensure we are done writing out all batches
        self.writer_pool.close()
        self.writer_pool.join()

        # Then write out the batch index.
        InferenceDataSetWriter.save_batch_index(self.result_dir, self.all_ids, self.all_batch_nums)

    @staticmethod
    def save_batch_index(result_dir: Path, all_ids: np.ndarray, all_batch_nums: np.ndarray):
        """Save a batch index in the result directory provided

        Parameters
        ----------
        result_dir : Path
            The results directory
        all_ids : np.ndarray
            All IDs to write out.
        all_batch_nums : np.ndarray
            The corresponding batch numbers for the IDs provided.
        """
        batch_index_dtype = np.dtype([("id", np.int64), ("batch_num", np.int64)])
        batch_index = np.zeros(len(all_ids), batch_index_dtype)
        batch_index["id"] = np.array(all_ids)
        batch_index["batch_num"] = np.array(all_batch_nums)
        batch_index.sort(order="id")

        filename = "batch_index.npy"
        savepath = result_dir / filename
        if savepath.exists():
            RuntimeError("The path to save batch index already exists.")
        np.save(savepath, batch_index, allow_pickle=False)

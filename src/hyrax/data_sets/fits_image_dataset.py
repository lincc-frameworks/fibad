# ruff: noqa: D101, D102

import logging
import time
from collections.abc import Generator, Iterable, Iterator
from concurrent.futures import Executor
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Optional, Union

import numpy as np
import numpy.typing as npt
from astropy.io import fits
from astropy.table import Table
from torch import Tensor, from_numpy
from torch.utils.data import Dataset
from torchvision.transforms.v2 import CenterCrop, Compose, Lambda, Transform

from hyrax.config_utils import ConfigDict

from .data_set_registry import HyraxDataset

logger = logging.getLogger(__name__)

files_dict = dict[str, dict[str, str]]


class FitsImageDataSet(HyraxDataset, Dataset):
    _called_from_test = False

    def __init__(self, config: ConfigDict):
        self._config = config

        transform_str = config["data_set"]["transform"]
        self.use_cache = config["data_set"]["use_cache"]

        if transform_str:
            transform_func = self._get_np_function(transform_str)
            self.transform = Lambda(lambd=transform_func)
        else:
            self.transform = None

        self._init_from_path(config["general"]["data_dir"])

        # Relies on self.filters_ref and self.filter_catalog_table which are both determined
        # inside _init_from_path()
        logger.debug("Preparing Metadata")
        metadata = self._prepare_metadata()
        super().__init__(config, metadata)

        self._before_preload()

        if config["data_set"]["preload_cache"] and self.use_cache:
            self.preload_thread = Thread(
                name=f"{self.__class__.__name__}-preload-tensor-cache",
                daemon=True,
                # Note we are passing only the function and self explicitly to the thread
                # This ensures the current object is in shared thread memory.
                target=self._preload_tensor_cache.__func__,  # type: ignore[attr-defined]
                args=(self,),
            )
            self.preload_thread.start()

    def _init_from_path(self, path: Union[Path, str]):
        """__init__ helper. Initialize an HSC data set from a path. This involves several filesystem scan
        operations and will ultimately open and read the header info of every fits file in the given directory

        Parameters
        ----------
        path : Union[Path, str]
            Path or string specifying the directory path that is the root of all filenames in the
            catalog table
        """
        self.path = path

        # This is common code
        filter_catalog = None
        if self.config["data_set"]["filter_catalog"]:
            filter_catalog = Path(self.config["data_set"]["filter_catalog"])

        self.filter_catalog_table = self._read_filter_catalog(filter_catalog)
        if self.filter_catalog_table is None:
            # xcxc write a warning about you don't get metadata if you didn't supply
            # a table
            pass

        self.files = self._parse_filter_catalog(self.filter_catalog_table)
        if self.files is None:
            # xcxc rewrite this error to be more informative to a user
            # xcxc Check if files is essentially the right types
            raise RuntimeError("xcxc Cannot continue without files. Probably a subclass messed it up.")

        first_filter_dict = next(iter(self.files.values()))
        self.num_filters = len(first_filter_dict)

        crop_transform = self._set_crop_transform()
        self.transform = (
            Compose([crop_transform, self.transform]) if self.transform is not None else crop_transform
        )

        self.tensors: dict[str, Tensor] = {}
        self.tensorboard_start_ns = time.monotonic_ns()
        self.tensorboardx_logger = None

        logger.info(f"FitsImageDataSet has {len(self)} objects")

    def _set_crop_transform(self) -> Transform:
        """
        Returns the crop transform on the image

        If overriden, subclass must:
        1) set self.cutout_shape to a tuple of ints representing the size of the cutouts that will be
        returned at some point in the init flow.

        2) Return the crop transform only so it can be added to the transform stack appropriately.
        """
        self.cutout_shape = self.config["data_set"]["crop_to"] if self.config["data_set"]["crop_to"] else None

        if not isinstance(self.cutout_shape, list) or not len(self.cutout_shape) == 2:
            raise RuntimeError("No cutout shape provided")
            # xcxc better error for user about the cutout size

        return CenterCrop(size=self.cutout_shape)

    def _read_filter_catalog(self, filter_catalog_path: Optional[Path]) -> Optional[Table]:
        if filter_catalog_path is None:
            return None

        if not filter_catalog_path.exists():
            logger.error(f"Filter catalog file {filter_catalog_path} given in config does not exist.")
            return None

        table = Table.read(filter_catalog_path, format="fits")
        colnames = table.colnames

        if "object_id" not in colnames:
            logger.error(f"Filter catalog file {filter_catalog_path} has no column object_id")
            return None

        table.add_index("object_id")
        table.add_index("filter")

        # xcxc we may want a warning here about required columns so FitsImageDataSet can function
        return table

    def _parse_filter_catalog(self, table: Optional[Table]) -> None:
        """Sets self.files by parsing the catalog.

        Subclasses may override this function to control parsing of the table more directly, but the
        overriding class must create the files dict which has type dict[object_id -> dict[filter -> filename]]
        with object_id, filter, and filename all strings.  In the case of no filter distinction, a single
        flag value may be used for the filter dict keys in the inner dicts.

        Parameters
        ----------
        table : Table
            The catalog we read in

        """
        # xcxc warn/error on various malformed table issues?
        filter_catalog: files_dict = {}

        for row in table:
            object_id = str(row["object_id"])
            filter = row["filter"]
            filename = row["filename"]

            # Insert into the filter catalog.
            if object_id not in filter_catalog:
                filter_catalog[object_id] = {}
            filter_catalog[object_id][filter] = filename

        return filter_catalog

    def _get_np_function(self, transform_str: str) -> Callable[..., Any]:
        """
        _get_np_function. Returns the numpy mathematical function that the
        supplied string maps to; or raises an error if the supplied string
        cannot be mapped to a function.

        Parameters
        ----------
        transform_str: str
            The string to me mapped to a numpy function
        """

        try:
            func: Callable[..., Any] = getattr(np, transform_str)
            if callable(func):
                return func
        except AttributeError as err:
            msg = f"{transform_str} is not a valid numpy function.\n"
            msg += "The string passed to the transform variable needs to be a numpy function"
            raise RuntimeError(msg) from err

    def _before_preload(self) -> None:
        # Provided so subclasses can make edits to the class after full initialization
        # but before the cache preload thread starts iterating over the datastructure and
        # fetching
        pass

    # xcxc read this later to see if it still works in init flow
    def _prepare_metadata(self) -> Optional[Table]:
        # This happens when filter_catalog_table is injected in unit tests
        if FitsImageDataSet._called_from_test:
            return None

        if self.filter_catalog_table is None:
            return None

        # Get all object_ids in enumeration order
        sorted_object_ids = np.array([int(id) for id in self.ids()])

        # Filter for the reference filter
        mask = self.filter_catalog_table["filter"] == self.filters_ref[0]
        filter_catalog_table_dedup = self.filter_catalog_table[mask]

        # Build fast lookup from object_id to row index
        id_to_index = {oid: i for i, oid in enumerate(filter_catalog_table_dedup["object_id"])}

        # Extract rows in the desired order
        try:
            row_indices = [id_to_index[oid] for oid in sorted_object_ids]
        except KeyError as e:
            missing_id = e.args[0]
            logger.error(f"Object ID {missing_id} not found in filtered metadata table.")
            raise

        metadata = filter_catalog_table_dedup[row_indices]

        # Filter for the appropriate columns
        colnames = list(self.filter_catalog_table.colnames)
        colnames.remove("filename")
        colnames.remove("filter")

        logger.debug("Finished preparing metadata")
        return metadata[colnames]

    def shape(self) -> tuple[int, int, int]:
        """Shape of the individual cutouts this will give to a model

        Returns
        -------
        tuple[int,int,int]
            Tuple describing the dimensions of the 3 dimensional tensor handed back to models
            The first index is the number of filters
            The second index is the width of each image
            The third index is the height of each image
        """
        return (self.num_filters, self.cutout_shape[0], self.cutout_shape[1])

    def __len__(self) -> int:
        """Returns number of objects in this loader

        Returns
        -------
        int
            number of objects in this data loader
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> Tensor:
        if idx >= len(self.files) or idx < 0:
            raise IndexError

        # Use the list of object IDs for explicit indexing
        object_id = list(self.files.keys())[idx]

        return self._object_id_to_tensor(object_id)

    def __contains__(self, object_id: str) -> bool:
        """Allows you to do `object_id in dataset` queries. Used by testing code.

        Parameters
        ----------
        object_id : str
            The object ID you'd like to know if is in the dataset

        Returns
        -------
        bool
            True of the object_id given is in the data set
        """
        return object_id in list(self.files.keys())

    def _get_file(self, index: int) -> Path:
        """Private indexing method across all files.

        Returns the file path corresponding to the given index.

        The index is zero-based and defined in the same manner as the total order of _all_files() and
        _object_files() iterator. Useful if you have an np.array() or list built from _all_files() and you
        need to select an individual item.

        Only valid after self.object_ids, self.files, self.path, and self.num_filters have been initialized
        in __init__

        Parameters
        ----------
        index : int
            Index, see above for order semantics

        Returns
        -------
        Path
            The path to the file
        """
        object_index = int(index / self.num_filters)
        object_id = list(self.files.keys())[object_index]
        filters = self.files[object_id]
        filter_names = sorted(list(filters))
        filter = filter_names[index % self.num_filters]
        return self._file_to_path(filters[filter])

    def ids(self, log_every=None) -> Generator[str]:
        """Public read-only iterator over all object_ids that enforces a strict total order across
        objects. Will not work prior to self.files initialization in __init__

        Yields
        ------
        Iterator[str]
            Object IDs currently in the dataset
        """
        log = log_every is not None and isinstance(log_every, int)
        for index, object_id in enumerate(self.files):
            if log and index != 0 and index % log_every == 0:
                logger.info(f"Processed {index} objects")
            yield str(object_id)
        else:
            if log:
                logger.info(f"Processed {index} objects")

    def _all_files(self):
        """
        Private read-only iterator over all files that enforces a strict total order across
        objects and filters. Will not work prior to self.files, and self.path initialization in __init__

        Yields
        ------
        Path
            The path to the file.
        """
        for object_id in self.ids():
            for filename in self._object_files(object_id):
                yield filename

    def _filter_filename(self, object_id):
        """
        Private read-only iterator over all files for a given object. This enforces a strict total order
        across filters. Will not work prior to self.files initialization in __init__

        Yields
        ------
        filter_name, file name
            The name of a filter and the file name for the fits file.
            The file name is relative to self.path
        """
        filters = self.files[object_id]
        filter_names = sorted(list(filters))
        for filter_name in filter_names:
            yield filter_name, filters[filter_name]

    def _object_files(self, object_id):
        """
        Private read-only iterator over all files for a given object. This enforces a strict total order
        across filters. Will not work prior to self.files, and self.path initialization in __init__

        Yields
        ------
        Path
            The path to the file.
        """
        for _, filename in self._filter_filename(object_id):
            yield self._file_to_path(filename)

    def _file_to_path(self, filename: str) -> Path:
        """Turns a filename into a full path suitable for open. Equivalent to:

        `Path(self.path) / Path(filename)`

        Parameters
        ----------
        filename : str
            The filename string

        Returns
        -------
        Path
            A full path that is openable.
        """
        return Path(self.path) / Path(filename)

    @staticmethod
    def _determine_numprocs_preload():
        # This is hardcoded to a reasonable value for hyak
        # TODO: Unify this function and _determine_numprocs(). Ideally we would have
        # either a multiprocessing.Pool or concurrent.futures.Executor interface taking
        # an i/o bound callable which returns a number of bytes read. This reusable
        # component would titrate the number of worker threads/processes to achieve a
        # maximal throughput as measured by returns from the callable vs wall-clock time.
        return 50

    def _preload_tensor_cache(self):
        """
        When preloading the tensor cache is configured, this is called on a separate thread by __init__()
        to perform a preload of every tensor in the dataset.
        """
        from concurrent.futures import ThreadPoolExecutor

        logger.info("Preloading FitsImageDataSet cache...")

        with ThreadPoolExecutor(max_workers=FitsImageDataSet._determine_numprocs_preload()) as executor:
            tensors = self._lazy_map_executor(executor, self.ids(log_every=1_000_000))

            start_time = time.monotonic_ns()
            for idx, (id, tensor) in enumerate(zip(self.ids(), tensors)):
                self.tensors[id] = tensor

                # Output timing every 1k tensors
                if idx % 1_000 == 0 and idx != 0:
                    self._log_duration_tensorboard("preload_1k_obj_s", start_time)
                    start_time = time.monotonic_ns()

    def _lazy_map_executor(self, executor: Executor, ids: Iterable[str]) -> Iterator[Tensor]:
        """This is a version of concurrent.futures.Executor map() which lazily evaluates the iterator passed
        We do this because we do not want all of the tensors to remain in memory during pre-loading. We would
        prefer a smaller set of in-flight tensors.

        The total number of in progress jobs is set at FitsImageDataSet._determine_numprocs().

        The total number of tensors is slightly greater than that owing to out-of-order execution.

        This approach was copied from:
        https://gist.github.com/CallumJHays/0841c5fdb7b2774d2a0b9b8233689761

        Parameters
        ----------
        executor : concurrent.futures.Executor
            An executour for running our futures
        work_fn : Callable[[str], torch.Tensor]
            The function that makes tensors out of object_ids
        ids : Iterable[str]
            An iterable list of object IDs.

        Yields
        ------
        Iterator[torch.Tensor]
            An iterator over torch tensors, lazily loaded by running the work_fn as needed.
        """

        from concurrent.futures import FIRST_COMPLETED, Future, wait

        max_futures = FitsImageDataSet._determine_numprocs_preload()
        queue: list[Future[Tensor]] = []
        in_progress: set[Future[Tensor]] = set()
        ids_iter = iter(ids)

        try:
            while True:
                for _ in range(max_futures - len(in_progress)):
                    id = next(ids_iter)
                    future = executor.submit(self._read_object_id.__func__, self, id)  # type: ignore[attr-defined]
                    queue.append(future)
                    in_progress.add(future)

                _, in_progress = wait(in_progress, return_when=FIRST_COMPLETED)

                while queue and queue[0].done():
                    yield queue.pop(0).result()

        except StopIteration:
            wait(queue)
            for future in queue:
                try:
                    result = future.result()
                except Exception as e:
                    raise e
                else:
                    yield result

    def _log_duration_tensorboard(self, name: str, start_time: int):
        """Log a duration to tensorboardX. NOOP if no tensorboard logger configured

        The time logged is a floating point number of seconds derived from integer
        monotonic nanosecond measurements. time.monotonic_ns() is used for the current time

        The step number for the scalar series is an integer number of microseonds.

        Parameters
        ----------
        name : str
            The name of the scalar to log to tensorboard
        start_time : int
            integer number of nanoseconds. Should be from time.monotonic_ns() when the duration started

        """
        now = time.monotonic_ns()
        name = f"{self.__class__.__name__}/" + name
        if self.tensorboardx_logger:
            since_tensorboard_start_us = (start_time - self.tensorboard_start_ns) / 1.0e3

            duration_s = (now - start_time) / 1.0e9
            self.tensorboardx_logger.add_scalar(name, duration_s, since_tensorboard_start_us)

    def _check_object_id_to_tensor_cache(self, object_id: str) -> Optional[Tensor]:
        return self.tensors.get(object_id, None)

    def _populate_object_id_to_tensor_cache(self, object_id: str) -> Tensor:
        data_torch = self._read_object_id(object_id)
        self.tensors[object_id] = data_torch
        return data_torch

    def _read_object_id(self, object_id: str) -> Tensor:
        start_time = time.monotonic_ns()

        # Read all the files corresponding to this object
        data = []

        for filepath in self._object_files_filters_ref(object_id):
            file_start_time = time.monotonic_ns()
            raw_data = fits.getdata(filepath, memmap=False)
            data.append(raw_data)
            self._log_duration_tensorboard("file_read_time_s", file_start_time)

        self._log_duration_tensorboard("object_read_time_s", start_time)

        data_torch = self._convert_to_torch(data)
        self._log_duration_tensorboard("object_total_read_time_s", start_time)
        return data_torch

    def _convert_to_torch(self, data: list[npt.ArrayLike]) -> Tensor:
        start_time = time.monotonic_ns()

        # Push all the filter data into a tensor object
        data_np = np.array(data)
        data_torch = from_numpy(data_np.astype(np.float32))

        # Apply our transform stack
        data_torch = self.transform(data_torch) if self.transform is not None else data_torch

        self._log_duration_tensorboard("object_convert_tensor_time_s", start_time)
        return data_torch

    # TODO: Performance Change when files are read/cache pytorch tensors?
    #
    # This function loads from a file every time __getitem__ is called
    # Do we want to pre-cache these into memory in init?
    # Do we want to memoize them on first __getitem__ call?
    #
    # For now we just do it the naive way
    def _object_id_to_tensor(self, object_id: str) -> Tensor:
        """Converts an object_id to a pytorch tensor with dimenstions (self.num_filters, self.cutout_shape[0],
        self.cutout_shape[1]). This is done by reading the file and slicing away any excess pixels at the
        far corners of the image from (0,0).

        The current implementation reads the files once the first time they are accessed, and then
        keeps them in a dict for future accesses.

        Parameters
        ----------
        object_id : str
            The object_id requested

        Returns
        -------
        torch.Tensor
            A tensor with dimension (self.num_filters, self.cutout_shape[0], self.cutout_shape[1])
        """
        start_time = time.monotonic_ns()

        if self.use_cache is False:
            return self._read_object_id(object_id)

        data_torch = self._check_object_id_to_tensor_cache(object_id)
        if data_torch is not None:
            self._log_duration_tensorboard("cache_hit_s", start_time)
            return data_torch

        data_torch = self._populate_object_id_to_tensor_cache(object_id)
        self._log_duration_tensorboard("cache_miss_s", start_time)
        return data_torch

# ruff: noqa: D101, D102

import logging
import re
from pathlib import Path
from typing import Union

import numpy as np
import torch
from astropy.io import fits
from torch.utils.data import Dataset
from torchvision.transforms.v2 import CenterCrop, Compose, Lambda

from .data_loader_registry import fibad_data_loader

logger = logging.getLogger(__name__)


@fibad_data_loader
class HSCDataLoader:
    def __init__(self, data_loader_config):
        self.config = data_loader_config
        self._data_set = self.data_set()

    def get_data_loader(self):
        """This is the primary method for this class.

        Returns
        -------
        torch.utils.data.DataLoader
            The dataloader to use for training.
        """
        return self.data_loader(self.data_set())

    def data_set(self):
        # Only construct a data set once per loader object, since it involves a filesystem scan.
        if self.__dict__.get("_data_set", None) is not None:
            return self._data_set

        self.config.get("path", "./data")

        # TODO: What will be a reasonable set of tranformations?
        # For now tanh all the values so they end up in [-1,1]
        # Another option might be sinh, but we'd need to mess with the example autoencoder module
        # Because it goes from unbounded NN output space -> [-1,1] with tanh in its decode step.
        transform = Lambda(lambd=np.tanh)

        return HSCDataSet(self.config.get("path", "./data"), transform=transform)

    def data_loader(self, data_set):
        return torch.utils.data.DataLoader(
            data_set,
            batch_size=self.config.get("batch_size", 4),
            shuffle=self.config.get("shuffle", True),
            num_workers=self.config.get("num_workers", 2),
        )

    def shape(self):
        return self.data_set().shape()


class HSCDataSet(Dataset):
    def __init__(self, path: Union[Path, str], transform=None):
        """Initialize an HSC data set from a path. This involves several filesystem scan operations and will
        ultimately open and read the header info of every fits file in the given directory

        Parameters
        ----------
        path : Union[Path, str]
            Path or string specifying the directory path to scan. It is expected that all files will
            be flat in this directory
        transform : _type_, optional
            _description_, by default None
        """
        self.path = path
        self.transform = transform

        self.files = self._scan_files()

        # We choose the first file in the dict as the prototypical set of filters
        # Any objects lacking this full set of filters will be pruned by
        # _prune_objects
        filters_ref = list(list(self.files.values())[0])

        self.num_filters = len(filters_ref)

        self.object_ids = self._prune_objects(self.files, filters_ref)

        self.cutout_width, self.cutout_height = self._check_file_dimensions()

        # Set up our default transform to center-crop the image to the common size before
        # Applying any transforms we were passed.
        crop = CenterCrop(size=(self.cutout_width, self.cutout_height))
        self.transform = Compose([crop, self.transform]) if self.transform is not None else crop

        self.tensors = {}

        logger.info(f"HSC Data set loader has {len(self)} objects")

    def _scan_files(self) -> dict[str, dict[str, str]]:
        """Class initialization helper

        Returns
        -------
        dict[str,dict[str,str]]
            Nested dictionary where the first level maps object_id -> dict, and the second level maps
            filter_name -> file name. Corresponds to self.files
        """
        files = {}
        # Go scan the path for object ID's so we have a list.
        for filepath in Path(self.path).glob("[0-9]*.fits"):
            filename = filepath.name
            m = re.match(r"([0-9]{17})_.*\_(HSC-[GRIZY]).fits", filename)
            object_id = m[1]
            filter = m[2]

            if files.get(object_id) is None:
                files[object_id] = {}

            files[object_id][filter] = filename

        return files

    def _prune_objects(self, files: dict[str, dict[str, str]], filters_ref: list[str]) -> list[str]:
        """Class initialization helper. Prunes files dict (which will be self.files). Removes any objects
        which do not ahve all the filters specified in filters_ref

        Parameters
        ----------
        files : dict[str,dict[str,str]]
            Nested dictionary where the first level maps object_id -> dict, and the second level maps
            filter_name -> file name. This is created by _scan_files()

        filters_ref : list[str]
            List of the filter names

        Returns
        -------
        list[str]
            List of all object IDs which survived the prune.
        """
        filters_ref = sorted(filters_ref)
        prune_count = 0
        for object_id, filters in list(files.items()):
            filters = sorted(list(filters))
            if filters != filters_ref:
                logger.warning(
                    f"HSCDataSet in {self.path} has the wrong group of filters for object {object_id}."
                )
                logger.warning(f"Dropping object {object_id} from the dataset.")
                logger.info(f"Filters for object {object_id} were {filters}")
                logger.debug(f"Reference filters were {filters_ref}")
                prune_count += 1
                # Remove any object IDs for which we don't have all the filters
                del files[object_id]

        # Dump all object IDs into a list so there is an explicit indexing/ordering convention
        # valid for the lifetime of this object.
        object_ids = list(files)

        # Log about the pruning process
        pre_prune_object_count = len(object_ids) + prune_count
        prune_fraction = prune_count / pre_prune_object_count
        if prune_fraction > 0.05:
            logger.error("Greater than 5% of objects in the data directory were pruned.")
        elif prune_fraction > 0.01:
            logger.warning("Greater than 1% of objects in the data directory were pruned.")
        logger.info(f"Pruned {prune_count} out of {pre_prune_object_count} objects")

        return object_ids

    def _check_file_dimensions(self) -> tuple[int, int]:
        """Class initialization helper. Scan all files to determine the minimal pixel size of images

        It is assumed that all the cutouts will be of very similar size; however, HSC's cutout
        server does not return exactly the same number of pixels for every query, even when it
        is given the same angular spread for every cutout.

        Machine learning models expect all images to be the same size.

        This function warns on significant differences (>2px) on any dimension between the largest
        and smallest images.

        Returns
        -------
        tuple(int,int)
            The minimum width and height in pixels of the entire dataset. In other words: the maximal image
            size in pixels that can be generated from ALL cutout images via cropping.
        """
        all_widths, all_heights = ([], [])

        for filepath in self._all_files():
            with fits.open(filepath) as hdul:
                width, height = hdul[1].shape
            all_widths.append(width)
            all_heights.append(height)

        cutout_width = np.min(all_widths)
        cutout_height = np.min(all_heights)

        if (
            np.abs(cutout_width - np.mean(all_widths)) > 1
            or np.abs(cutout_height - np.mean(all_heights)) > 1
            or np.abs(np.max(all_widths) - np.mean(all_widths)) > 1
            or np.abs(np.max(all_heights) - np.mean(all_heights)) > 1
        ):
            logger.warning("Some images differ from the mean width or height of all images by more than 1px")
            logger.warning(f"Images will be cropped to ({cutout_width}px, {cutout_height}px)")
            min_width_file = self._get_file(np.argmin(all_widths))
            logger.warning(f"See {min_width_file} for an example image of width {cutout_width}px")
            min_height_file = self._get_file(np.argmin(all_heights))
            logger.warning(f"See {min_height_file} for an example image of height {cutout_height}px")

        return cutout_width, cutout_height

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
        return (self.num_filters, self.cutout_width, self.cutout_height)

    def __len__(self) -> int:
        """Returns number of objects in this loader

        Returns
        -------
        int
            number of objects in this data loader
        """
        return len(self.object_ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx >= len(self.object_ids) or idx < 0:
            raise IndexError

        # Use the list of object IDs for explicit indexing
        object_id = self.object_ids[idx]

        tensor = self._object_id_to_tensor(object_id)

        return tensor

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
        object_id = self.object_ids[int(index / self.num_filters)]
        filters = self.files[object_id]
        filter_names = sorted(list(filters))
        filter = filter_names[index % self.num_filters]
        return self._file_to_path(filters[filter])

    def _all_files(self) -> Path:
        """
        Private read-only iterator over all files that enforces a strict total order across
        objects and filters. Will not work prior to self.object_ids, self.files, and self.path
        initialization in __init__

        Yields
        ------
        Path
            The path to the file.
        """
        for object_id in self.object_ids:
            for filename in self._object_files(object_id):
                yield filename

    def _object_files(self, object_id) -> Path:
        """
        Private read-only iterator over all files for a given object. This enforces a strict total order
        across filters. Will not work prior to self.object_ids, self.files, and self.path initialization
        in __init__

        Yields
        ------
        Path
            The path to the file.
        """
        filters = self.files[object_id]
        filter_names = sorted(list(filters))
        for filter in filter_names:
            yield self._file_to_path(filters[filter])

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

    # TODO: Performance Change when files are read/cache pytorch tensors?
    #
    # This function loads from a file every time __getitem__ is called
    # Do we want to pre-cache these into memory in init?
    # Do we want to memoize them on first __getitem__ call?
    #
    # For now we just do it the naive way
    def _object_id_to_tensor(self, object_id: str) -> torch.Tensor:
        """Converts an object_id to a pytorch tensor with dimenstions (self.num_filters, self.cutout_width,
        self.cutout_height). This is done by reading the file and slicing away any excess pixels at the
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
            A tensor with dimension (self.num_filters, self.cutout_width, self.cutout_height)
        """
        data_torch = self.tensors.get(object_id, None)
        if data_torch is not None:
            return data_torch

        # Read all the files corresponding to this object
        data = []

        for filepath in self._object_files(object_id):
            raw_data = fits.getdata(filepath, memmap=False)
            data.append(raw_data)

        # Push all the filter data into a tensor object
        data_np = np.array(data)
        data_torch = torch.from_numpy(data_np.astype(np.float32))

        # Apply our transform stack
        data_torch = self.transform(data_torch) if self.transform is not None else data_torch

        self.tensors[object_id] = data_torch
        return data_torch

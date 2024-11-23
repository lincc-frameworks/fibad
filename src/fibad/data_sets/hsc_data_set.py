# ruff: noqa: D101, D102

import logging
import multiprocessing
import os
import re
import resource
from copy import copy, deepcopy
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch
from astropy.io import fits
from astropy.table import Table
from schwimmbad import MultiPool
from torch.utils.data import Dataset
from torchvision.transforms.v2 import CenterCrop, Compose, Lambda

from fibad.download import Downloader
from fibad.downloadCutout.downloadCutout import (
    parse_bool,
    parse_degree,
    parse_latitude,
    parse_longitude,
    parse_rerun,
    parse_tract_opt,
    parse_type,
)

from .data_set_registry import fibad_data_set

logger = logging.getLogger(__name__)


@fibad_data_set
class HSCDataSet(Dataset):
    _called_from_test = False

    """Interface object to allow simple access to splits on a corpus of HSC data files

    f/s operations and management are handled in HSCDatSetContainer
    splits on the dataset and their generation are handled by HSCDataSetSplit

    """

    def __init__(self, config, split: Union[str, None]):
        # initialize the filesystem references
        self.container = HSCDataSetContainer(config)

        # initalize our splits from configuration
        self._create_splits(config)

        # Set the split to what was requested.
        self._set_split(split)

    def _create_splits(self, config):
        seed = config["prepare"]["seed"] if config["prepare"]["seed"] else None

        # Init the splits based on config values
        train_size = config["prepare"]["train_size"] if config["prepare"]["train_size"] else None
        test_size = config["prepare"]["test_size"] if config["prepare"]["test_size"] else None
        validate_size = config["prepare"]["validate_size"] if config["prepare"]["validate_size"] else None

        # Convert all values specified as counts into ratios of the underlying container
        if isinstance(train_size, int):
            train_size = train_size / len(self.container)
        if isinstance(test_size, int):
            test_size = test_size / len(self.container)
        if isinstance(validate_size, int):
            validate_size = validate_size / len(self.container)

        # Initialize Test size when not provided
        if test_size is None:
            if train_size is None:
                train_size = 0.25

            if validate_size is None:  # noqa: SIM108
                test_size = 1.0 - train_size
            else:
                test_size = 1.0 - (train_size + validate_size)

        # Initialize train size when not provided, and can be inferred from test_size and validate_size.
        if train_size is None:
            if validate_size is None:  # noqa: SIM108
                train_size = 1.0 - test_size
            else:
                train_size = 1.0 - (test_size + validate_size)

        # If we still don't have a validate size, decide whether we will infer a validate size
        if (validate_size is None) and (np.round(train_size + test_size) != 1.0):
            validate_size = 1.0 - (train_size + test_size)

        # If splits cover more than the entire dataset, error out.
        if validate_size is None:
            if np.round(train_size + test_size) > 1.0:
                raise RuntimeError("Split fractions add up to more than 1.0")
        elif np.round(train_size + test_size + validate_size) > 1.0:
            raise RuntimeError("Split fractions add up to more than 1.0")

        # Generate splits
        self.splits = {}
        self.splits["test"] = HSCDataSetSplit(self.container, test_size, seed=seed)
        rest = copy(self.splits["test"]).complement()
        self.splits["train"] = HSCDataSetSplit(rest, train_size, seed=seed)

        # Validate is only generated if it is provided, or if both test and train are provided.
        if validate_size:
            rest = rest.logical_and(copy(self.splits["train"]).complement())
            self.splits["validate"] = HSCDataSetSplit(rest, validate_size, seed=seed)

        logger.info("HSC Data Set Splits loaded are:")
        for key, value in self.splits.items():
            logger.info(f"{key} split contains {len(value)} items")

    def _set_split(self, split: Union[str, None] = None):
        self.current_split = self.splits.get(split, self.container)

        if split is not None and self.current_split == self.container:
            splits = list(self.splits.keys())
            raise RuntimeError(f"Split {split} does not exist. valid split names are {splits}")

    def shape(self) -> tuple[int, int, int]:
        return self.container.shape()

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.current_split[idx]

    def __len__(self) -> int:
        return len(self.current_split)

    def rebuild_manifest(self, config):
        return self.container._rebuild_manifest(config)


class HSCDataSetSplit(Dataset):
    def __init__(
        self,
        data: Union["HSCDataSetContainer", "HSCDataSetSplit"],
        ratio: float,
        seed: Union[int, None] = None,
    ):
        """
        This class represents a split of an HSCDataset.

        It should only get created by passing in an existing HSCDataSetContainer (or HSCDataSetSplit)
        and splitting it according to the train_test_split like parameters. When you split a split,
        all splits end up referring to the same uderlying HSCDataSetContainer object.

        Each encodes a subset of the underlying HSCDataSetContainer by keeping a list of boolean values.

        Parameters
        ----------
        data : Union[HSCDataSetContainer, &quot;HSCDataSetSplit&quot;]
            The underlying HSCDataSet or split to operate on. Creating a split from an existing split ends up
            referring to a subset of the data selected by the original split, but the new object only refers
            to an underlying HSCDataSet object, not any other split object.
        ratio : float
            Ratio of the underlying data source to use for this split. This is expressed as a fraction of the
            HSCDataSetContainer even when an HSCDataSetSplit is passed.
        seed : Union[int, None] , optional
            The seed value to provide to the random number generator, or None if you would like to use system
            entropy to generate a seed. None by default.
        shuffle : bool, optional
            Whether to shuffle the order of the underlying data when accessing the split object, by default
            True
        """
        self.rng = np.random.default_rng(seed)

        if ratio > 1.0 or ratio < 0.0:
            msg = f"Split provided for HSCDatSetSplit as a ratio is {ratio}, which is not between 0.0 and 1.0"
            raise RuntimeError(msg)

        self.data = data.data if isinstance(data, HSCDataSetSplit) else data

        # The length of this split once constructed
        length = int(np.round(len(self.data) * ratio))

        if isinstance(data, HSCDataSetSplit):
            # If we're splitting a split we need to modify the existing mask of the prior split
            # Namely we switch some true values to false to more of the underlying dataset
            split = data
            self.mask = copy(split.mask)
            remove_count = len(split) - length
            self._flip_mask_values(remove_count, "true_to_false")

        else:
            # If we're splitting a normal hscdataset we generate a single mask with the appropriate values
            self.mask = np.zeros(len(data), dtype=bool)
            self._flip_mask_values(length, "false_to_true")

        self.indexes = np.nonzero(self.mask)[0]

    def _flip_mask_values(self, num: int, mode: Literal["false_to_true", "true_to_false"]):
        """
        Private helper to flips some values of self.mask. The direction to flip is controlled by the
        mode parameter. Either the function randomly finds `num` true values to flip to false, or `num` false
        values to flip to true.

        This function is used during object construction to create a set number of randomly selected true
        values in the mask.

        Parameters
        ----------
        num : int
            The number of values to flip
        mode : Literal[&quot;false_to_true&quot;, &quot;true_to_false&quot;]
            The mode to work in, either flipping True values false or the reverse

        Raises
        ------
        RuntimeError
            It is a RuntimeError to try to flip more values than the mask has of that type.

        """
        mask_tmp = np.logical_not(self.mask) if mode == "false_to_true" else self.mask
        target_val = mode == "false_to_true"
        target_indexes = np.nonzero(mask_tmp)[0]

        if num > len(target_indexes):
            msg_mode = mode.replace("_", " ")
            num_tgt = len(target_indexes)
            msg = f"Cannot flip {num} values {msg_mode} when only {num_tgt} {target_val} values exist in mask"
            raise RuntimeError(msg)

        change_indexes = self.rng.permutation(target_indexes)[:num]
        for i in change_indexes:
            self.mask[i] = target_val

    def complement(self) -> "HSCDataSetSplit":
        """Mutates the split by inverting it with respect to the underlying dataset.

        e.g. if you have an underlying dataset with 5 members, and indexes 1,2, and 4 are part of this split
        The compliment would be a dataset selecting indexes 0 and 3.
        """
        self.mask = np.logical_not(self.mask)
        self.indexes = np.nonzero(self.mask)[0]
        return self

    def logical_and(self, obj: "HSCDataSetSplit") -> "HSCDataSetSplit":
        """Takes the logical and of this object and the passed in object. self is modified, the passed in
        object is not

        If the self object selects indicies 1,2 and 4 and the passed in object selects indicies 2, 4, and 0
        the self object would be modified to select indicies 2, and 4 only.

        It is a RuntimeError to and two split objects that do not reference the same underlying HSCDataSet

        Parameters
        ----------
        obj : HSCDataSetSplit
            The object to and with
        """
        if self.data != obj.data:
            msg = "Tried to take logical and of two HSCDataSetSplits with different HSCDataSet objects"
            raise RuntimeError(msg)

        self.mask = np.logical_and(self.mask, obj.mask)
        self.indexes = np.nonzero(self.mask)[0]
        return self

    def __copy__(self) -> "HSCDataSetSplit":
        # Create a HSCDataSetSplit with no data selected, but the same data source as self
        copy_object = HSCDataSetSplit(self.data, 0.0)

        # Copy mask and indexes over
        copy_object.mask = self.mask.copy()
        copy_object.indexes = self.indexes.copy()

        # Copy RNG state over.
        copy_object.rng = deepcopy(self.rng)

        return copy_object

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[self.indexes[idx]]


dim_dict = dict[str, tuple[int, int]]
files_dict = dict[str, dict[str, str]]


class HSCDataSetContainer(Dataset):
    def __init__(self, config):
        # TODO: What will be a reasonable set of tranformations?
        # For now tanh all the values so they end up in [-1,1]
        # Another option might be sinh, but we'd need to mess with the example autoencoder module
        # Because it goes from unbounded NN output space -> [-1,1] with tanh in its decode step.
        transform = Lambda(lambd=np.tanh)

        crop_to = config["data_set"]["crop_to"]
        filters = config["data_set"]["filters"]

        if config["data_set"]["filter_catalog"]:
            filter_catalog = Path(config["data_set"]["filter_catalog"])
        elif not config.get("rebuild_manifest", False):
            # Note "rebuild_manifest" is not a config, its a hack for rebuild_manifest mode
            # to ensure we don't use the manifest we believe is corrupt.
            filter_catalog = Path(config["general"]["data_dir"]) / Downloader.MANIFEST_FILE_NAME
            if not filter_catalog.exists():
                filter_catalog = False
        else:
            filter_catalog = False

        self._init_from_path(
            config["general"]["data_dir"],
            transform=transform,
            cutout_shape=crop_to if crop_to else None,
            filters=filters if filters else None,
            filter_catalog=Path(filter_catalog) if filter_catalog else None,
        )

    def _init_from_path(
        self,
        path: Union[Path, str],
        *,
        transform=None,
        cutout_shape: Optional[tuple[int, int]] = None,
        filters: Optional[list[str]] = None,
        filter_catalog: Optional[Path] = None,
    ):
        """__init__ helper. Initialize an HSC data set from a path. This involves several filesystem scan
        operations and will ultimately open and read the header info of every fits file in the given directory

        Parameters
        ----------
        path : Union[Path, str]
            Path or string specifying the directory path to scan. It is expected that all files will
            be flat in this directory
        transform : torchvision.transforms.v2.Transform, optional
            Transformation to apply to every image in the dataset, by default None
        cutout_shape: tuple[int,int], optional
            Forces all cutouts to be a particular pixel size. If this size is larger than the pixel dimension
            of particular cutouts on the filesystem, those objects are dropped from the data set.
        filters: list[str], optional
            Forces all cutout tensors provided to be from the list of HSC filters provided. If provided, any
            cutouts which do not have fits files corresponding to every filter in the list will be dropped
            from the data set. Defaults to None. If not provided, the filters available on the filesystem for
            the first object in the directory will be used.
        filter_catalog: Path, optional
            Path to a .fits file which specifies objects and or files to use directly, bypassing the default
            of attempting to use every file in the path.
            Columns for this fits file are object_id (required), filter (optional), filename (optional), and
            dims (optional tuple of x/y pixel size of images).
             - Filenames must be relative to the path provided to this function.
             - When filters and filenames are both provided, initialization skips a directory listing, which
               can provide better performance on large datasets.
             - When filters, filenames, and dims are specified we also skip opening the files to get
               the dimensions. This can also provide better performance on large datasets.
        """
        self.path = path
        self.transform = transform

        self.filter_catalog = self._read_filter_catalog(filter_catalog)
        if isinstance(self.filter_catalog, tuple):
            self.files = self.filter_catalog[0]
            self.dims = self.filter_catalog[1]
        elif isinstance(self.filter_catalog, dict):
            self.files = self.filter_catalog
            self.dims = self._scan_file_dimensions()
        else:
            self.files = self._scan_file_names(filters)
            self.dims = self._scan_file_dimensions()

        # If no filters provided, we choose the first file in the dict as the prototypical set of filters
        # Any objects lacking this full set of filters will be pruned by _prune_objects
        filters_ref = list(list(self.files.values())[0]) if filters is None else filters

        self.num_filters = len(filters_ref)

        self.cutout_shape = cutout_shape

        self.pruned_objects = {}
        self._prune_objects(filters_ref)

        if self.cutout_shape is None:
            self.cutout_shape = self._check_file_dimensions()

        # Set up our default transform to center-crop the image to the common size before
        # Applying any transforms we were passed.
        crop = CenterCrop(size=self.cutout_shape)
        self.transform = Compose([crop, self.transform]) if self.transform is not None else crop

        self.tensors = {}

        logger.info(f"HSC Data set loader has {len(self)} objects")

    def _scan_file_names(self, filters: Optional[list[str]] = None) -> files_dict:
        """Class initialization helper

        Parameters
        ----------
        filters : list[str], optional
            If passed, only these filters will be scanned for from the data files. Defaults to None, which
            corresponds to the standard set of filters ["HSC-G","HSC-R","HSC-I","HSC-Z","HSC-Y"].

        Returns
        -------
        dict[str,dict[str,str]]
            Nested dictionary where the first level maps object_id -> dict, and the second level maps
            filter_name -> file name. Corresponds to self.files
        """

        logger.info(f"Scanning files in directory {self.path}")

        object_id_regex = r"[0-9]{17}"
        filter_regex = r"HSC-[GRIZY]" if filters is None else "|".join(filters)
        full_regex = f"({object_id_regex})_.*_({filter_regex}).fits"

        files = {}
        # Go scan the path for object ID's so we have a list.
        for index, filepath in enumerate(Path(self.path).iterdir()):
            filename = filepath.name

            # If we are filtering based off a user-provided catalog of object ids, Filter out any
            # objects_ids not in the catalog. Do this before regex match for speed of discarding
            # irrelevant files.
            if isinstance(self.filter_catalog, list) and filename[:17] not in self.filter_catalog:
                continue

            m = re.match(full_regex, filename)
            # Skip files that don't allow us to extract both object_id and filter
            if m is None:
                continue

            object_id = m[1]
            filter = m[2]

            if files.get(object_id) is None:
                files[object_id] = {}

            if files[object_id].get(filter) is None:
                files[object_id][filter] = filename
            else:
                msg = f"Duplicate object ID {object_id} detected.\n"
                msg += f"File {filename} conflicts with already scanned file {files[object_id][filter]} "
                msg += "and will not be included in the data set."
                logger.error(msg)

            if index != 0 and index % 1_000_000 == 0:
                logger.info(f"Processed {index} files.")
        else:
            logger.info(f"Processed {index+1} files")

        return files

    def _read_filter_catalog(
        self, filter_catalog_path: Optional[Path]
    ) -> Optional[Union[list[str], files_dict, tuple[files_dict, dim_dict]]]:
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

        # We are dealing with just a list of object_ids
        if "filter" not in colnames and "filename" not in colnames:
            return list(table["object_id"])

        # Or a table that lacks both filter and filename
        elif "filter" not in colnames or "filename" not in colnames:
            msg = f"Filter catalog file {filter_catalog_path} provides one of filters or filenames "
            msg += "without the other. Filesystem scan will still occur without both defined."
            logger.warning(msg)
            return list(set(table["object_id"]))

        # We have filter and filename defined so we can assemble the catalog at file level.
        filter_catalog = {}
        if "dim" in colnames:
            dim_catalog = {}

        for row in table:
            object_id = row["object_id"]
            filter = row["filter"]
            filename = row["filename"]

            if object_id not in filter_catalog:
                filter_catalog[object_id] = {}

            filter_catalog[object_id][filter] = filename

            # Dimension is optional
            if "dim" in colnames:
                if object_id not in dim_catalog:
                    dim_catalog[object_id] = []
                dim_catalog[object_id].append(tuple(row["dim"]))

        return (filter_catalog, dim_catalog) if "dim" in colnames else filter_catalog

    @staticmethod
    def _determine_numprocs() -> int:
        # Figure out how many CPUs we are allowed to use
        cpu_count = None
        sched_getaffinity = getattr(os, "sched_getaffinity", None)

        if sched_getaffinity:
            cpu_count = len(sched_getaffinity(0))
        elif multiprocessing:
            cpu_count = multiprocessing.cpu_count()
        else:
            cpu_count = 1

        # Ideally we would use ~75 processes per CPU to attempt to saturate
        # I/O bandwidth using a small number of CPUs.
        numproc = 1 if HSCDataSet._called_from_test else 75 * cpu_count
        numproc = HSCDataSetContainer._fixup_limit(
            numproc,
            resource.RLIMIT_NOFILE,
            lambda proc: int(4 * proc + 10),
            lambda nofile: int((nofile - 10) / 4),
        )

        numproc = HSCDataSetContainer._fixup_limit(
            numproc, resource.RLIMIT_NPROC, lambda proc: proc, lambda proc: proc
        )
        return numproc

    @staticmethod
    def _fixup_limit(nproc, res, est_limit, est_procs) -> int:
        # If launching this many processes would trigger other resource limits, work around them
        limit_soft, limit_hard = resource.getrlimit(res)

        # If we would violate the hard limit, calculate the number of processes that wouldn't
        # violate the limit
        if limit_hard < est_limit(nproc):
            nproc = est_procs(limit_hard)

        # If we would violate the soft limit, attempt to change it, leaving the hard limit alone
        try:
            if limit_soft < est_limit(nproc):
                resource.setrlimit(res, (est_limit(nproc), limit_hard))
        finally:
            # If the change doesn't take, then reduce the number of processes again
            limit_soft, limit_hard = resource.getrlimit(res)
            if limit_soft < est_limit(nproc):
                nproc = est_procs(limit_soft)

        return nproc

    def _scan_file_dimensions(self) -> dim_dict:
        # Scan the filesystem to get the widths and heights of all images into a dict
        logger.info("Scanning for dimensions...")

        retval = {}
        with MultiPool(processes=HSCDataSetContainer._determine_numprocs()) as pool:
            args = (
                (object_id, list(self._object_files(object_id)))
                for object_id in self.ids(log_every=1_000_000)
            )
            retval = dict(pool.imap(self._scan_file_dimension, args, chunksize=1000))
        return retval

    @staticmethod
    def _scan_file_dimension(processing_unit: tuple[str, list[str]]) -> list[tuple[int, int]]:
        object_id, filenames = processing_unit
        return (object_id, [HSCDataSetContainer._fits_file_dims(filepath) for filepath in filenames])

    @staticmethod
    def _fits_file_dims(filepath):
        try:
            with fits.open(filepath) as hdul:
                return hdul[1].shape
        except OSError:
            return (0, 0)

    def _prune_objects(self, filters_ref: list[str]):
        """Class initialization helper. Prunes objects from the list of objects.

        1) Removes any objects which do not have all the filters specified in filters_ref
        2) If a cutout_shape was provided in the constructor, prunes files that are too small
           for the chosen cutout size

        This function deletes from self.files and self.dims via _prune_object

        Parameters
        ----------
        files : dict[str,dict[str,str]]
            Nested dictionary where the first level maps object_id -> dict, and the second level maps
            filter_name -> file name. This is created by _scan_files()

        filters_ref : list[str]
            List of the filter names

        """
        filters_ref = sorted(filters_ref)
        self.prune_count = 0
        for index, (object_id, filters) in enumerate(self.files.items()):
            # Drop objects with missing filters
            filters = sorted(list(filters))
            if filters != filters_ref:
                msg = f"HSCDataSet in {self.path} has the wrong group of filters for object {object_id}."
                self._mark_for_prune(object_id, msg)
                logger.info(f"Filters for object {object_id} were {filters}")
                logger.debug(f"Reference filters were {filters_ref}")

            # Drop objects that can't meet the coutout size provided
            elif self.cutout_shape is not None:
                for shape in self.dims[object_id]:
                    if shape[0] < self.cutout_shape[0] or shape[1] < self.cutout_shape[1]:
                        msg = f"A file for object {object_id} has shape ({shape[1]}px, {shape[1]}px)"
                        msg += " this is too small for the given cutout size of "
                        msg += f"({self.cutout_shape[0]}px, {self.cutout_shape[1]}px)"
                        self._mark_for_prune(object_id, msg)
                        break
            if index != 0 and index % 1_000_000 == 0:
                logger.info(f"Processed {index} objects for pruning")
        else:
            logger.info(f"Processed {index + 1} objects for pruning")

        # Prune marked objects
        for object_id, reason in self.pruned_objects.items():
            self._prune_object(object_id, reason)

        # Log about the pruning process
        pre_prune_object_count = len(self.files) + self.prune_count
        prune_fraction = self.prune_count / pre_prune_object_count
        if prune_fraction > 0.05:
            logger.error("Greater than 5% of objects in the data directory were pruned.")
        elif prune_fraction > 0.01:
            logger.warning("Greater than 1% of objects in the data directory were pruned.")
        logger.info(f"Pruned {self.prune_count} out of {pre_prune_object_count} objects")

    def _mark_for_prune(self, object_id, reason):
        self.pruned_objects[object_id] = reason

    def _prune_object(self, object_id, reason: str):
        logger.warning(reason)
        logger.warning(f"Dropping object {object_id} from the dataset")

        del self.files[object_id]
        del self.dims[object_id]
        self.prune_count += 1

    def _check_file_dimensions(self) -> tuple[int, int]:
        """Class initialization helper. Find the maximal pixel size that all images can support

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
        logger.info("Checking file dimensions to determine standard cutout size...")

        # Find the maximal cutout size that all images can support
        all_widths = [shape[0] for shape_list in self.dims.values() for shape in shape_list]
        all_heights = [shape[1] for shape_list in self.dims.values() for shape in shape_list]
        all_dimensions = all_widths + all_heights
        cutout_height = np.min(all_dimensions)
        cutout_width = cutout_height

        if (
            np.abs(cutout_width - np.mean(all_widths)) > 1
            or np.abs(cutout_height - np.mean(all_heights)) > 1
            or np.abs(np.max(all_widths) - np.mean(all_widths)) > 1
            or np.abs(np.max(all_heights) - np.mean(all_heights)) > 1
        ):
            msg = "Some images differ from the mean width or height of all images by more than 1px\n"
            msg += f"Images will be cropped to ({cutout_width}px, {cutout_height}px)\n"
            try:
                min_width_file = self._get_file(np.argmin(all_widths))
                min_height_file = self._get_file(np.argmin(all_heights))
                msg += f"See {min_width_file} for an example image of width {cutout_width}px\n"
                msg += f"See {min_height_file} for an example image of height {cutout_height}px"
            finally:
                logger.warning(msg)

        return cutout_width, cutout_height

    def _rebuild_manifest(self, config):
        if self.filter_catalog:
            raise RuntimeError("Cannot rebuild manifest. Set the filter_catalog=false and rerun")

        logger.info("Reading in catalog file... ")
        location_table = Downloader.filterfits(
            Path(config["download"]["fits_file"]).resolve(), ["object_id", "ra", "dec"]
        )

        obj_to_ra = {
            str(location_table["object_id"][index]): location_table["ra"][index]
            for index in range(len(location_table))
        }
        obj_to_dec = {
            str(location_table["object_id"][index]): location_table["dec"][index]
            for index in range(len(location_table))
        }

        del location_table

        logger.info("Assembling Manifest...")

        # These are the column names expected in a manifest file by the downloader
        column_names = Downloader.MANIFEST_COLUMN_NAMES
        columns = {column_name: [] for column_name in column_names}

        # These will vary every object and must be implemented below
        dynamic_column_names = ["object_id", "filter", "dim", "tract", "ra", "dec", "filename"]
        # These are pulled from config ("sw", "sh", "rerun", "type", "image", "mask", and "variance")
        static_column_names = [name for name in column_names if name not in dynamic_column_names]

        # Check that all column names we need for a manifest are either in static or dynamic columns
        for column_name in column_names:
            if column_name not in static_column_names and column_name not in dynamic_column_names:
                raise RuntimeError(f"Error Assembling manifest {column_name} not implemented")

        static_values = {
            "sw": parse_degree(config["download"]["sw"]),
            "sh": parse_degree(config["download"]["sh"]),
            "rerun": parse_rerun(config["download"]["rerun"]),
            "type": parse_type(config["download"]["type"]),
            "image": parse_bool(config["download"]["image"]),
            "mask": parse_bool(config["download"]["mask"]),
            "variance": parse_bool(config["download"]["variance"]),
        }

        for index, (object_id, filter, filename, dim) in enumerate(self._all_files_full()):
            for static_col in static_column_names:
                columns[static_col].append(static_values[static_col])

            for dynamic_col in dynamic_column_names:
                if dynamic_col == "object_id":
                    columns[dynamic_col].append(int(object_id))
                elif dynamic_col == "filter":
                    columns[dynamic_col].append(filter)
                elif dynamic_col == "dim":
                    columns[dynamic_col].append(dim)
                elif dynamic_col == "tract":
                    # There's value in pulling tract from the filename rather than the download catalog
                    # in case The catalog had it wrong, the filename will have the value the cutout server
                    # provided.
                    tract = filename.split("_")[4]
                    columns[dynamic_col].append(parse_tract_opt(tract))
                elif dynamic_col == "ra":
                    ra = obj_to_ra[object_id]
                    columns[dynamic_col].append(parse_longitude(ra))
                elif dynamic_col == "dec":
                    dec = obj_to_dec[object_id]
                    columns[dynamic_col].append(parse_latitude(dec))
                elif dynamic_col == "filename":
                    columns[dynamic_col].append(filename)
                else:
                    # The tower of if statements has been entirely to create this failure path.
                    # which will be hit when someone alters dynamic column names above without also
                    # writing an implementation.
                    raise RuntimeError(f"No implementation to process column {dynamic_col}")
            if index != 0 and index % 1_000_000 == 0:
                logger.info(f"Addeed {index} objects to manifest")
        else:
            logger.info(f"Addeed {index+1} objects to manifest")

        logger.info("Writing rebuilt manifest...")
        manifest_table = Table(columns)
        rebuilt_manifest_path = Path(config["general"]["data_dir"]) / "rebuilt_manifest.fits"
        manifest_table.write(rebuilt_manifest_path, overwrite=True, format="fits")

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

    def __getitem__(self, idx: int) -> torch.Tensor:
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
        return object_id in list(self.files.keys()) and object_id in list(self.dims.keys())

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

    def ids(self, log_every=None):
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
            yield object_id
        else:
            if log:
                logger.info(f"Processed {index} objects")

    def _all_files_full(self):
        """
        Private read-only iterator over all files that enforces a strict total order across
        objects and filters. Will not work prior to self.files, and self.path initialization in __init__

        Yields
        ------
        Tuple[object_id, filter, filename, dim]
            Members of this tuple are
            - The object_id as a string
            - The filter name as a string
            - The filename relative to self.path
            - A tuple containing the dimensions of the fits file in pixels.
        """
        for object_id in self.ids():
            dims = self.dims[object_id]
            for idx, (filter, filename) in enumerate(self._filter_filename(object_id)):
                yield (object_id, filter, filename, dims[idx])

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

    # TODO: Performance Change when files are read/cache pytorch tensors?
    #
    # This function loads from a file every time __getitem__ is called
    # Do we want to pre-cache these into memory in init?
    # Do we want to memoize them on first __getitem__ call?
    #
    # For now we just do it the naive way
    def _object_id_to_tensor(self, object_id: str) -> torch.Tensor:
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

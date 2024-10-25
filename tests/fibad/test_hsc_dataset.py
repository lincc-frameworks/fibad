import logging
import unittest.mock as mock
from copy import copy
from pathlib import Path

import numpy as np
import pytest
from fibad.data_sets.hsc_data_set import HSCDataSet, HSCDataSetSplit

test_dir = Path(__file__).parent / "test_data" / "dataloader"


class FakeFitsFS:
    """
    Mocks the only operations on a directory full of fits files that
    the dataloader should care about for initialization:

    1) Globbing fits files with Path.glob

    2) Reading the shape of the first table out of the fits header using
    astropy.io.fits.open

    If we end up doing anything more sophisticated this will need to be changed
    We are planning to scan 1M - 10M fits files so its unlikely we want to do
    more filesystem operations without a really good reason.
    """

    def __init__(self, test_files: dict):
        self.patchers = []

        self.test_files = test_files

        mock_paths = [Path(x) for x in list(test_files.keys())]
        target = "fibad.data_sets.hsc_data_set.Path.glob"
        self.patchers.append(mock.patch(target, return_value=mock_paths))

        mock_fits_open = mock.Mock(side_effect=self._open_file)
        self.patchers.append(mock.patch("astropy.io.fits.open", mock_fits_open))

    def _open_file(self, filename: Path, **kwargs) -> mock.Mock:
        shape = self.test_files[filename.name]
        mock_open_ctx = mock.Mock()
        mock_open_ctx.__enter__ = mock.Mock(return_value=["", np.zeros(shape)])
        mock_open_ctx.__exit__ = mock.Mock()
        return mock_open_ctx

    def __enter__(self):
        for patcher in self.patchers:
            patcher.start()

    def __exit__(self, *exc):
        for patcher in self.patchers:
            patcher.stop()


def mkconfig(crop_to=False, filters=False, train_size=0.2, test_size=0.6, validate_size=0, seed=False):
    """Makes a configuration that points at nonexistent path so HSCDataSet.__init__ will create an object,
    and our FakeFitsFS shim can be called.
    """
    return {
        "general": {"data_dir": "thispathdoesnotexist"},
        "data_set": {
            "crop_to": crop_to,
            "filters": filters,
        },
        "prepare": {
            "seed": seed,
            "train_size": train_size,
            "test_size": test_size,
            "validate_size": validate_size,
        },
    }


def generate_files(
    num_objects=10, num_filters=5, shape=(100, 100), offset=0, infill_str="all_filters"
) -> dict:
    """Generates a dictionary to pass in to FakeFitsFS.

    This generates a dict from filename->shape tuple for a set of uniform fake fits files
    corresponding to the naming convention used by the HSC data loader.

    Completely uniform data sets with all filters for all files with every file the same
    size are generated by this function. Sequential object_ids are used starting from 0
    by default.

    Parameters
    ----------
    num_objects : int, optional
        How many objects are represented in the data set, by default 10
    num_filters : int, optional
        How many filters does each object have, by default 5. If you provide a number greater than 5,
        only 5 filters will be output.
    shape : tuple, optional
        What are the dimensions of the image in each fits file, by default (100,100)
    offset : int, optional
        What is the first object_id to start with, by default 0
    infill_str: str, optional
        What to put in the fake filename in between the object ID and filter name. By default "all_filters"

    Returns
    -------
    dict
        Dictionary from filename -> shape appropriate for FakeFitsFS
    """
    filters = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"][:num_filters]
    test_files = {}
    for object_id in range(offset, num_objects + offset):
        for filter in filters:
            test_files[f"{object_id:017d}_{infill_str}_{filter}.fits"] = shape

    return test_files


def test_load(caplog):
    """Test to ensure loading a perfectly regular set of files works"""
    caplog.set_level(logging.WARNING)
    test_files = generate_files(num_objects=10, num_filters=5, shape=(262, 263))
    with FakeFitsFS(test_files):
        a = HSCDataSet(mkconfig(), split=None)

        # 10 objects should load
        assert len(a) == 10

        # The number of filters, and image dimensions should be correct
        assert a.shape() == (5, 262, 263)

        # No warnings should be printed
        assert caplog.text == ""


def test_load_duplicate(caplog):
    """Test to ensure duplicate fits files that reference the same object id and filter create the
    appropriate error messages.
    """
    caplog.set_level(logging.ERROR)
    test_files = generate_files(num_objects=10, num_filters=5, shape=(262, 263))
    duplicate_files = generate_files(num_objects=10, num_filters=5, shape=(262, 263), infill_str="duplicate")
    test_files.update(duplicate_files)
    with FakeFitsFS(test_files):
        a = HSCDataSet(mkconfig(), split=None)

        # Only 10 objects should load
        assert len(a) == 10

        # The number of filters, and image dimensions should be correct
        assert a.shape() == (5, 262, 263)

        # We should get duplicate object errors
        assert "Duplicate object ID" in caplog.text

        # We should get errors that include the duplicate filenames
        assert "_duplicate_" in caplog.text

        # The duplicate files should not be in the data set
        for filepath in a.container._all_files():
            assert "_duplicate_" not in str(filepath)


def test_prune_warn_1_percent(caplog):
    """Test to ensure when >1% of loaded objects are missing a filter, that is a warning
    and that the resulting dataset drops the objects that are missing filters
    """
    caplog.set_level(logging.WARNING)

    # Generate two files which
    test_files = generate_files(num_objects=98, num_filters=2, shape=(100, 100))
    # Object 101 is missing the HSC-G filter, we only provide the R filter
    test_files["00000000000000101_missing_g_HSC-R.fits"] = (100, 100)

    with FakeFitsFS(test_files):
        a = HSCDataSet(mkconfig(), split=None)

        # We should have the correct number of objects
        assert len(a) == 98

        # Object 2 should not be loaded
        assert "00000000000000101" not in a.container

        # We should Error log because greater than 5% of the objects were pruned
        assert "Greater than 1% of objects in the data directory were pruned." in caplog.text

        # We should warn that we dropped an object explicitly
        assert "Dropping object" in caplog.text


def test_prune_error_5_percent(caplog):
    """Test to ensure when >5% of loaded objects are missing a filter, that is an error
    and that the resulting dataset drops the objects that are missing filters
    """
    caplog.set_level(logging.ERROR)

    # Generate two files which
    test_files = generate_files(num_objects=18, num_filters=2, shape=(100, 100))
    # Object 20 is missing the HSC-G filter, we only provide the R filter
    test_files["00000000000000020_missing_g_HSC-R.fits"] = (100, 100)

    with FakeFitsFS(test_files):
        a = HSCDataSet(mkconfig(), split=None)

        # We should have two objects, having dropped one.
        assert len(a) == 18

        # Object 20 should not be loaded
        assert "00000000000000020" not in a.container

        # We should Error log because greater than 5% of the objects were pruned
        assert "Greater than 5% of objects in the data directory were pruned." in caplog.text


def test_crop(caplog):
    """Test to ensure that in the presence of heterogenous sizes within 1px of a central size
    We load all images and crop to the smallest dimenensions without any logs
    """
    caplog.set_level(logging.WARNING)
    test_files = {}
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(100, 100), offset=0))
    # Add some images with dimensions 1 px larger
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(101, 100), offset=10))
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(100, 101), offset=20))
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(101, 101), offset=30))
    # Add some images with dimensions 1 px smaller
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(99, 100), offset=40))
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(100, 99), offset=50))
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(99, 99), offset=60))

    with FakeFitsFS(test_files):
        a = HSCDataSet(mkconfig(), split=None)

        assert len(a) == 70
        assert a.shape() == (5, 99, 99)

        # No warnings should be printed since we're within 1px of the mean size
        assert caplog.text == ""


def test_crop_warn_2px_larger(caplog):
    """Test to ensure that in the presence of heterogenous sizes within 2px of a central size
    We load all images and crop to the smallest dimenensions and warn the user of the issue
    """
    caplog.set_level(logging.WARNING)
    test_files = {}
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(100, 100), offset=0))
    # Add some images with dimensions 2 px larger
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(102, 100), offset=10))
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(100, 102), offset=20))
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(102, 102), offset=30))
    # Add some images with dimensions 1 px smaller
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(99, 100), offset=40))
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(100, 99), offset=50))
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(99, 99), offset=60))

    with FakeFitsFS(test_files):
        a = HSCDataSet(mkconfig(), split=None)

        assert len(a) == 70
        assert a.shape() == (5, 99, 99)

        # We should warn that images differ
        assert "Some images differ" in caplog.text


def test_crop_warn_2px_smaller(caplog):
    """Test to ensure that in the presence of heterogenous sizes within 2px of a central size
    We load all images and crop to the smallest dimenensions and warn the user of the issue
    """
    caplog.set_level(logging.WARNING)
    test_files = {}
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(100, 100), offset=0))
    # Add some images with dimensions 1 px larger
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(101, 100), offset=10))
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(100, 101), offset=20))
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(101, 101), offset=30))
    # Add some images with dimensions 2 px smaller
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(98, 100), offset=40))
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(100, 98), offset=50))
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(98, 98), offset=60))

    with FakeFitsFS(test_files):
        a = HSCDataSet(mkconfig(), split=None)

        assert len(a) == 70
        assert a.shape() == (5, 98, 98)

        # We should warn that images differ
        assert "Some images differ" in caplog.text


def test_prune_size(caplog):
    """Test to ensure images that are too small will be pruned from the data set when a custom size is
    passed."""
    caplog.set_level(logging.WARNING)
    test_files = {}
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(100, 100), offset=0))
    # Add some images with dimensions 1 px larger
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(101, 101), offset=20))
    # Add some images with dimensions 2 px smaller
    test_files.update(generate_files(num_objects=10, num_filters=5, shape=(98, 98), offset=30))

    with FakeFitsFS(test_files):
        a = HSCDataSet(mkconfig(crop_to=(99, 99)), split=None)

        assert len(a) == 20
        assert a.shape() == (5, 99, 99)

        # We should warn that we are dropping objects and the reason
        assert "Dropping object" in caplog.text
        assert "too small" in caplog.text


def test_partial_filter(caplog):
    """Test to ensure when we only load some of the filters, only those filters end up in the dataset"""
    caplog.set_level(logging.WARNING)
    test_files = generate_files(num_objects=10, num_filters=5, shape=(262, 263))
    with FakeFitsFS(test_files):
        a = HSCDataSet(mkconfig(filters=["HSC-G", "HSC-R"]), split=None)

        # 10 objects should load
        assert len(a) == 10

        # The number of filters, and image dimensions should be correct
        assert a.shape() == (2, 262, 263)

        # No warnings should be printed
        assert caplog.text == ""


def test_partial_filter_prune_warn_1_percent(caplog):
    """Test to ensure when a the user supplies a filter list and >1% of loaded objects are
    missing a filter, that is a warning and that the resulting dataset drops the objects that
    are missing filters.
    """
    caplog.set_level(logging.WARNING)

    # Generate two files which
    test_files = generate_files(num_objects=98, num_filters=3, shape=(100, 100))
    # Object 101 is missing the HSC-G and HSC-I filters, we only provide the R filter
    test_files["00000000000000101_missing_g_HSC-R.fits"] = (100, 100)

    with FakeFitsFS(test_files):
        a = HSCDataSet(mkconfig(filters=["HSC-R", "HSC-I"]), split=None)

        # We should have the correct number of objects
        assert len(a) == 98

        # Object 101 should not be loaded
        assert "00000000000000101" not in a.container

        # We should Error log because greater than 5% of the objects were pruned
        assert "Greater than 1% of objects in the data directory were pruned." in caplog.text

        # We should warn that we dropped an object explicitly
        assert "Dropping object" in caplog.text


def test_split():
    """Test splitting in the default config where train, test, and validate are all specified"""
    test_files = generate_files(num_objects=100, num_filters=3, shape=(100, 100))
    with FakeFitsFS(test_files):
        a = HSCDataSet(mkconfig(filters=["HSC-G", "HSC-R", "HSC-I"]), split="validate")
        assert len(a) == 20

        a = HSCDataSet(mkconfig(filters=["HSC-G", "HSC-R", "HSC-I"]), split="test")
        assert len(a) == 60

        a = HSCDataSet(mkconfig(filters=["HSC-G", "HSC-R", "HSC-I"]), split="train")
        assert len(a) == 20


def test_split_no_validate():
    """Test splitting when validate is overridden"""
    test_files = generate_files(num_objects=100, num_filters=3, shape=(100, 100))
    with FakeFitsFS(test_files):
        config = mkconfig(filters=["HSC-G", "HSC-R", "HSC-I"], validate_size=False)

        a = HSCDataSet(config, split="test")
        assert len(a) == 60

        a = HSCDataSet(config, split="train")
        assert len(a) == 20

        a = HSCDataSet(config, split="validate")
        assert len(a) == 20


def test_split_no_validate_no_test():
    """Test splitting when validate and test are overridden"""
    test_files = generate_files(num_objects=100, num_filters=3, shape=(100, 100))
    with FakeFitsFS(test_files):
        config = mkconfig(filters=["HSC-G", "HSC-R", "HSC-I"], validate_size=False, test_size=False)

        a = HSCDataSet(config, split="test")
        assert len(a) == 80

        a = HSCDataSet(config, split="train")
        assert len(a) == 20

        with pytest.raises(RuntimeError):
            a = HSCDataSet(config, split="validate")


def test_split_no_validate_no_train():
    """Test splitting when validate and train are overridden"""
    test_files = generate_files(num_objects=100, num_filters=3, shape=(100, 100))
    with FakeFitsFS(test_files):
        config = mkconfig(filters=["HSC-G", "HSC-R", "HSC-I"], validate_size=False, train_size=False)

        a = HSCDataSet(config, split="test")
        assert len(a) == 60

        a = HSCDataSet(config, split="train")
        assert len(a) == 40

        with pytest.raises(RuntimeError):
            a = HSCDataSet(config, split="validate")


def test_split_invalid_ratio():
    """Test that split RuntimeErrors when provided with an invalid ratio"""
    test_files = generate_files(num_objects=100, num_filters=3, shape=(100, 100))
    with FakeFitsFS(test_files):
        config = mkconfig(filters=["HSC-G", "HSC-R", "HSC-I"], validate_size=False, train_size=1.1)
        with pytest.raises(RuntimeError):
            HSCDataSet(config, split=None)

        config = mkconfig(filters=["HSC-G", "HSC-R", "HSC-I"], validate_size=False, train_size=-0.1)
        with pytest.raises(RuntimeError):
            HSCDataSet(config, split=None)


def test_split_no_splits_configured():
    """Test splitting when all splits are overriden, and nothing is specified."""
    test_files = generate_files(num_objects=100, num_filters=3, shape=(100, 100))
    with FakeFitsFS(test_files):
        config = mkconfig(
            filters=["HSC-G", "HSC-R", "HSC-I"], validate_size=False, test_size=False, train_size=False
        )

        a = HSCDataSet(config, split="test")
        assert len(a) == 75

        a = HSCDataSet(config, split="train")
        assert len(a) == 25

        with pytest.raises(RuntimeError):
            a = HSCDataSet(config, split="validate")


def test_split_values_configured():
    """Test splitting when all splits are integer data counts"""
    test_files = generate_files(num_objects=100, num_filters=3, shape=(100, 100))
    with FakeFitsFS(test_files):
        config = mkconfig(filters=["HSC-G", "HSC-R", "HSC-I"], validate_size=22, test_size=56, train_size=22)

        a = HSCDataSet(config, split="test")
        assert len(a) == 56

        a = HSCDataSet(config, split="train")
        assert len(a) == 22

        a = HSCDataSet(config, split="validate")
        assert len(a) == 22


def test_split_values_configured_no_validate():
    """Test splitting when all splits are integer data counts and validate is not configured
    so the total selected data doesn't cover the dataset.
    """
    test_files = generate_files(num_objects=100, num_filters=3, shape=(100, 100))
    with FakeFitsFS(test_files):
        config = mkconfig(filters=["HSC-G", "HSC-R", "HSC-I"], test_size=56, train_size=22)

        a = HSCDataSet(config, split="test")
        assert len(a) == 56

        a = HSCDataSet(config, split="train")
        assert len(a) == 22


def test_split_invalid_configured():
    """Test that split RuntimeErrors when provided with an invalid datapoint count"""
    test_files = generate_files(num_objects=100, num_filters=3, shape=(100, 100))
    with FakeFitsFS(test_files):
        config = mkconfig(filters=["HSC-G", "HSC-R", "HSC-I"], validate_size=False, train_size=120)
        with pytest.raises(RuntimeError):
            HSCDataSet(config, split=None)

        config = mkconfig(filters=["HSC-G", "HSC-R", "HSC-I"], validate_size=False, train_size=-10)
        with pytest.raises(RuntimeError):
            HSCDataSet(config, split=None)


def test_split_values_rng():
    """Generate twice with the same RNG seed, verify same values are selected."""
    test_files = generate_files(num_objects=100, num_filters=3, shape=(100, 100))
    with FakeFitsFS(test_files):
        config = mkconfig(filters=["HSC-G", "HSC-R", "HSC-I"], test_size=56, train_size=22, seed=5)

        a = HSCDataSet(config, split="test")
        b = HSCDataSet(config, split="test")

        assert all([a == b for a, b in zip(a.current_split.indexes, b.current_split.indexes)])
        assert a.current_split.rng.random() == b.current_split.rng.random()


def test_split_copy():
    """Generate a split, copy it, then verify they:
    - Both have the same underlying container object
    - That the rng for both returns the same next number
    - Both mask the same underlying data while having separate arrays.
    """
    test_files = generate_files(num_objects=100, num_filters=3, shape=(100, 100))
    with FakeFitsFS(test_files):
        config = mkconfig(filters=["HSC-G", "HSC-R", "HSC-I"], validate_size=22, test_size=56, train_size=22)

        testsplit = HSCDataSet(config, split="test").current_split
        copysplit = copy(testsplit)

        assert testsplit.data is copysplit.data
        assert testsplit.mask is not copysplit.mask
        assert all([a == b for a, b in zip(testsplit.mask, copysplit.mask)])
        assert testsplit.indexes is not copysplit.indexes
        assert all([a == b for a, b in zip(testsplit.indexes, copysplit.indexes)])
        assert testsplit.rng is not copysplit.rng
        assert testsplit.rng.random() == copysplit.rng.random()


def test_split_compliment():
    """Generate a split and its complement. Verify they are compliments"""
    test_files = generate_files(num_objects=100, num_filters=3, shape=(100, 100))
    with FakeFitsFS(test_files):
        config = mkconfig(filters=["HSC-G", "HSC-R", "HSC-I"], validate_size=22, test_size=56, train_size=22)

        testsplit = HSCDataSet(config, split="test").current_split
        complement = copy(testsplit).complement()

        assert all([a != b for a, b in zip(testsplit.mask, complement.mask)])


def test_split_and():
    """Generate two splits, and them together. Verify they are actually a conjunction"""
    test_files = generate_files(num_objects=100, num_filters=3, shape=(100, 100))
    with FakeFitsFS(test_files):
        config = mkconfig(filters=["HSC-G", "HSC-R", "HSC-I"], validate_size=False, test_size=False)

        dataset = HSCDataSet(config, split="test")
        test_split = dataset.current_split
        train_split = HSCDataSetSplit(dataset.container, ratio=0.5)

        and_split = train_split.logical_and(test_split)
        and_mask = np.logical_and(test_split.mask, train_split.mask)

        assert all([a == b for a, b in zip(and_split.mask, and_mask)])


def test_split_and_conflicting_datasets():
    """Generate two splits from different data sets, and them together. Verify this RuntimeErrors"""
    test_files = generate_files(num_objects=100, num_filters=3, shape=(100, 100))
    with FakeFitsFS(test_files):
        config = mkconfig(filters=["HSC-G", "HSC-R", "HSC-I"], validate_size=False, test_size=False)

        a = HSCDataSet(config, split="test")
        b = HSCDataSet(config, split="test")

        with pytest.raises(RuntimeError):
            a.current_split.logical_and(b.current_split)

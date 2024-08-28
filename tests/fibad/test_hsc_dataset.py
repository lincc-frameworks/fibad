import logging
import unittest.mock as mock
from pathlib import Path

import numpy as np
from fibad.data_loaders.hsc_data_loader import HSCDataSet

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
        target = "fibad.data_loaders.hsc_data_loader.Path.glob"
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


def generate_files(num_objects=10, num_filters=5, shape=(100, 100), offset=0) -> dict:
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

    Returns
    -------
    dict
        Dictionary from filename -> shape appropriate for FakeFitsFS
    """
    filters = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"][:num_filters]
    test_files = {}
    for object_id in range(offset, num_objects + offset):
        for filter in filters:
            test_files[f"{object_id:017d}_all_filters_{filter}.fits"] = shape

    return test_files


def test_load(caplog):
    """Test to ensure loading a perfectly regular set of files works"""
    caplog.set_level(logging.WARNING)
    test_files = generate_files(num_objects=10, num_filters=5, shape=(262, 263))
    with FakeFitsFS(test_files):
        a = HSCDataSet("thispathdoesnotexist")

        # 10 objects should load
        assert len(a) == 10

        # The number of filters, and image dimensions should be correct
        assert a.shape() == (5, 262, 263)

        # No warnings should be printed
        assert caplog.text == ""


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
        a = HSCDataSet("thispathdoesnotexist")

        # We should have the correct number of objects
        assert len(a) == 98

        # Object 2 should not be loaded
        assert "00000000000000101" not in a.object_ids

        # We should Error log because greater than 5% of the objects were pruned
        assert "Greater than 1% of objects in the data directory were pruned." in caplog.text


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
        a = HSCDataSet("thispathdoesnotexist")

        # We should have two objects, having dropped one.
        assert len(a) == 18

        # Object 20 should not be loaded
        assert "00000000000000020" not in a.object_ids

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
        a = HSCDataSet("thispathdoesnotexist")

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
        a = HSCDataSet("thispathdoesnotexist")

        assert len(a) == 70
        assert a.shape() == (5, 99, 99)

        # No warnings should be printed since we're within 1px of the mean size
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
        a = HSCDataSet("thispathdoesnotexist")

        assert len(a) == 70
        assert a.shape() == (5, 98, 98)

        # No warnings should be printed since we're within 1px of the mean size
        assert "Some images differ" in caplog.text
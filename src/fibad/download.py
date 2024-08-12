import contextlib
import os
from pathlib import Path
from typing import Union

from astropy.table import Table, hstack

import fibad.downloadCutout.downloadCutout as dC

# These are the fields that are allowed to vary across the locations
# input from the catalog fits file. Other values for HSC cutout server
# must be provided by config.
#
# Order here is intentional, this is also a sort order to optimize
# queries to the cutout server.
variable_fields = ["tract", "ra", "dec"]


@contextlib.contextmanager
def working_directory(path: Path):
    """
    Context Manager to change our working directory.

    Supports downloadCutouts which always writes to cwd.
    """
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def run(args, config):
    """
    Main entrypoint for downloading cutouts from HSC for use with fibad
    """

    config = config.get("download", {})

    print("Download command")

    # Filter the fits file for the fields we want
    column_names = ["object_id"] + variable_fields
    locations = filterfits(config.get("fits_file"), column_names)

    # Sort by tract, ra, dec to optimize speed that the cutout server can serve us
    #
    # TODO: See if this sort is performed by downloadCutouts
    # It appears downloadCutouts is doing some sorting prior to download, but
    # unclear if it is the same sort
    locations.sort(variable_fields)

    # TODO slice up the locations
    locations = locations[0:10]

    # make a list of rects
    rects = create_rects(locations, offset=0, default=rect_from_config(config))

    # Configure global parameters for the downloader
    dC.set_max_connections(num=config.get("max_connections", 2))

    # pass the rects to the cutout downloader
    download_cutout_group(
        rects=rects, cutout_dir=config.get("cutout_dir"), user=config["username"], password=config["password"]
    )

    print(locations)


# TODO add error checking
def filterfits(filename: str, column_names: list[str]) -> Table:
    """
    Read a fits file with the required column names for making cutouts

    Returns an astropy table containing only the necessary fields

    The easiest way to make one of these is to select from the main HSC catalog
    """
    t = Table.read(filename)
    columns = [t[column] for column in column_names]
    return hstack(columns, uniq_col_name="{table_name}", table_names=column_names)


def rect_from_config(config: dict) -> dC.Rect:
    """
    Takes our Download config and loads cutout config
    common to all cutouts into a prototypical Rect for downloading
    """
    return dC.Rect.create(
        sw=config["sw"],
        sh=config["sh"],
        filter=config["filter"],
        rerun=config["rerun"],
        type=config["type"],
    )


def create_rects(locations: Table, offset: int = 0, default: dC.Rect = None) -> list[dC.Rect]:
    """
    Create the rects we will need to pass to the downloader.
    One Rect per location in our list of sky locations.

    Rects are created with all fields in the default rect pre-filled

    Offset here is to allow multiple downloads on different sections of the source list
    without file name clobbering during the download phase. The offset is intended to be
    the index of the start of the locations table within some larger fits file.
    """
    rects = []
    for index, location in enumerate(locations):
        args = {field: location[field] for field in variable_fields}
        args["lineno"] = index + offset
        args["tract"] = str(args["tract"])
        rect = dC.Rect.create(default=default, **args)
        rects.append(rect)

    return rects


def download_cutout_group(rects: list[dC.Rect], cutout_dir: Union[str, Path], user, password):
    """
    Download cutouts to the given directory

    Calls downloadCutout.download, so supports long lists of rects and
    """
    with working_directory(Path(cutout_dir)):
        dC.download(rects, user=user, password=password, onmemory=False)

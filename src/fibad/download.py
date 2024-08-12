import contextlib
import datetime
import os
import urllib.request
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

    print("Download command Start")

    fits_file = config.get("fits_file", "")
    print(f"Reading in fits catalog: {fits_file}")
    # Filter the fits file for the fields we want
    column_names = ["object_id"] + variable_fields
    locations = filterfits(fits_file, column_names)

    # TODO slice up the locations to multiplex across connections if necessary, but for now
    # we simply mask off a few
    offset = config.get("offset", 0)
    end = offset + config.get("num_sources", 10)
    locations = locations[offset:end]

    # Make a list of rects to pass to downloadCutout
    rects = create_rects(locations, offset=0, default=rect_from_config(config))

    # Configure global parameters for the downloader
    dC.set_max_connections(num=config.get("max_connections", 2))

    print("Requesting cutouts")
    # pass the rects to the cutout downloader
    download_cutout_group(
        rects=rects, cutout_dir=config.get("cutout_dir"), user=config["username"], password=config["password"]
    )

    # print(locations)
    print("Done")


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


stats = {
    "request_duration": datetime.timedelta(),  # Time from request sent to first byte from the server
    "response_duration": datetime.timedelta(),  # Total time spent recieving and processing a response
    "request_size_bytes": 0,  # Total size of all requests
    "response_size_bytes": 0,  # Total size of all responses
    "snapshots": 0,  # Number of fits snapshots downloaded
}


def _stat_accumulate(name: str, value: Union[int, datetime.timedelta]):
    global stats
    stats[name] += value


def _print_stats():
    global stats

    total_dur_s = (stats["request_duration"] + stats["response_duration"]).total_seconds()

    resp_s = stats["response_duration"].total_seconds()
    down_rate_mb_s = (stats["response_size_bytes"] / (1024**2)) / resp_s

    req_s = stats["request_duration"].total_seconds()
    up_rate_mb_s = (stats["request_size_bytes"] / (1024**2)) / req_s

    snapshot_rate = stats["snapshots"] / total_dur_s

    print(
        f"Stats: Duration: {total_dur_s:.2f} s, Files: {stats['snapshots']}, \
Upload: {up_rate_mb_s:.2f} MB/s, Download: {down_rate_mb_s:.2f} MB/s File rate: {snapshot_rate:.2f} files/s",
        end="\r",
        flush=True,
    )


def request_hook(
    request: urllib.request.Request,
    request_start: datetime.datetime,
    response_start: datetime.datetime,
    response_size: int,
    chunk_size: int,
):
    """
    Called on each chunk of snapshots downloaded.
    Called immediately after the server has finished responding to the
    request, so datetime.datetime.now() is the end moment of the request

    request: Our request object
    request_start: datetime when we started sending the request
    response_start: when the server responded to the request
    response_size: Size in bytes of the response from the server.
    """

    now = datetime.datetime.now()

    _stat_accumulate("request_duration", response_start - request_start)
    _stat_accumulate("response_duration", now - response_start)
    _stat_accumulate("request_size_bytes", len(request.data))
    _stat_accumulate("response_size_bytes", response_size)
    _stat_accumulate("snapshots", chunk_size)

    _print_stats()


def download_cutout_group(rects: list[dC.Rect], cutout_dir: Union[str, Path], user, password):
    """
    Download cutouts to the given directory

    Calls downloadCutout.download, so supports long lists of rects and
    """
    with working_directory(Path(cutout_dir)):
        dC.download(rects, user=user, password=password, onmemory=True, request_hook=request_hook)

    print("")  # Print a newline so the stats stay and look pretty.

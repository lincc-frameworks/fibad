import contextlib
import datetime
import logging
import os
import urllib.request
from pathlib import Path
from typing import Union

import numpy as np
from astropy.table import Table, hstack

import fibad.downloadCutout.downloadCutout as dC

# These are the fields that are allowed to vary across the locations
# input from the catalog fits file. Other values for HSC cutout server
# must be provided by config.
variable_fields = ["tract", "ra", "dec"]

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def working_directory(path: Path):
    """
    Context Manager to change our working directory.
    Supports downloadCutouts which always writes to cwd.

    Parameters
    ----------
    path : Path
        Path that we change `Path.cwd()` while we are active.
    """
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def run(config):
    """
    Main entrypoint for downloading cutouts from HSC for use with fibad

    Parameters
    ----------
    config : dict
        Runtime configuration as a nested dictionary
    """

    config = config.get("download", {})

    logger.info("Download command Start")

    fits_file = config.get("fits_file", "")
    logger.info(f"Reading in fits catalog: {fits_file}")
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

    logger.info("Requesting cutouts")
    # pass the rects to the cutout downloader
    download_cutout_group(
        rects=rects,
        cutout_dir=config.get("cutout_dir"),
        user=config["username"],
        password=config["password"],
        retrywait=config.get("retry_wait", 30),
        retries=config.get("retries", 3),
        timeout=config.get("timeout", 3600),
        chunksize=config.get("chunk_size", 990),
    )

    logger.info("Done")


# TODO add error checking
def filterfits(filename: str, column_names: list[str]) -> Table:
    """Read a fits file with the required column names for making cutouts

    The easiest way to make such a fits file is to select from the main HSC catalog

    Parameters
    ----------
    filename : str
        The fits file to read in
    column_names : list[str]
        The columns that are filtered out

    Returns
    -------
    Table
       Returns an astropy table containing only the fields specified in column_names
    """
    t = Table.read(filename)
    columns = [t[column] for column in column_names]
    return hstack(columns, uniq_col_name="{table_name}", table_names=column_names)


def rect_from_config(config: dict) -> dC.Rect:
    """Takes our runtime config and loads cutout config
    common to all cutouts into a prototypical Rect for downloading

    Parameters
    ----------
    config : dict
        Runtime config, only the download section

    Returns
    -------
    dC.Rect
        A single rectangle with fields `sw`, `sh`, `filter`, `rerun`, and `type` populated from the config
    """
    return dC.Rect.create(
        sw=config["sw"],
        sh=config["sh"],
        filter=config["filter"],
        rerun=config["rerun"],
        type=config["type"],
    )


def create_rects(locations: Table, offset: int = 0, default: dC.Rect = None) -> list[dC.Rect]:
    """Create the rects we will need to pass to the downloader.
    One Rect per location in our list of sky locations.

    Rects are created with all fields in the default rect pre-filled

    Offset here is to allow multiple downloads on different sections of the source list
    without file name clobbering during the download phase. The offset is intended to be
    the index of the start of the locations table within some larger fits file.

    Parameters
    ----------
    locations : Table
        Table containing ra, dec locations in the sky
    offset : int, optional
        Index to start the `lineno` field in the rects at, by default 0. The purpose of this is to allow
        multiple downloads on different sections of a larger source list without file name clobbering during
        the download phase. This is important because `lineno` in a rect can becomes a file name parameter
        The offset is intended to be the index of the start of the locations table within some larger fits
        file.
    default : dC.Rect, optional
        The default Rect that contains properties common to all sky locations, by default None

    Returns
    -------
    list[dC.Rect]
        Rects populated with sky locations from the table
    """
    rects = []
    for index, location in enumerate(locations):
        args = {field: location[field] for field in variable_fields}
        args["lineno"] = index + offset
        args["tract"] = str(args["tract"])
        # Sets the file name on the rect to be the object_id, also includes other rect fields
        # which are interpolated at save time, and are native fields of dc.Rect.
        #
        # This name is also parsed by FailedChunkCollector.hook to identify the object_id, so don't
        # change it without updating code there too.
        args["name"] = f"{location['object_id']}_{{type}}_{{ra:.5f}}_{{dec:+.5f}}_{{tract}}_{{filter}}"
        rect = dC.Rect.create(default=default, **args)
        rects.append(rect)

    return rects


class DownloadStats:
    """Subsytem for keeping statistics on downloads:

    Accumulates time spent on request, responses as well as sizes for same and number of snapshots

    Can be used as a context manager for pretty printing.
    """

    def __init__(self):
        self.stats = {
            "request_duration": datetime.timedelta(),  # Time from request sent to first byte from the server
            "response_duration": datetime.timedelta(),  # Total time spent recieving and processing a response
            "request_size_bytes": 0,  # Total size of all requests
            "response_size_bytes": 0,  # Total size of all responses
            "snapshots": 0,  # Number of fits snapshots downloaded
        }

    def __enter__(self):
        return self.hook

    def __exit__(self, exc_type, exc_value, traceback):
        self._print_stats(logging.INFO)

    def _stat_accumulate(self, name: str, value: Union[int, datetime.timedelta]):
        """Accumulate a sum into the global stats dict

        Parameters
        ----------
        name : str
            Name of the stat. Assumed to exist in the dict already.
        value : Union[int, datetime.timedelta]
            How much time or count to add to the stat
        """
        self.stats[name] += value

    def _print_stats(self, log_level):
        """Print the accumulated stats including bandwidth calculated from duration and sizes

        This prints out multiple lines with `\r` at the end in order to create a continuously updating
        line of text during download if your terminal supports it.

        If you use this class as a context manager, the end of context will output a newline, perserving
        the last line of stats in your terminal
        """
        total_dur_s = (self.stats["request_duration"] + self.stats["response_duration"]).total_seconds()

        resp_s = self.stats["response_duration"].total_seconds()
        down_rate_mb_s = (self.stats["response_size_bytes"] / (1024**2)) / resp_s

        req_s = self.stats["request_duration"].total_seconds()
        up_rate_mb_s = (self.stats["request_size_bytes"] / (1024**2)) / req_s

        snapshot_rate = self.stats["snapshots"] / total_dur_s

        stats_message = f"Stats: Duration: {total_dur_s:.2f} s, "
        stats_message += f"Files: {self.stats['snapshots']}, "
        stats_message += f"Upload: {up_rate_mb_s:.2f} MB/s, "
        stats_message += f"Download: {down_rate_mb_s:.2f} MB/s, "
        stats_message += f"File rate: {snapshot_rate:.2f} files/s"

        logger.log(log_level, stats_message)

    def hook(
        self,
        request: urllib.request.Request,
        request_start: datetime.datetime,
        response_start: datetime.datetime,
        response_size: int,
        chunk_size: int,
    ):
        """This hook is called on each chunk of snapshots downloaded.
        It is called immediately after the server has finished responding to the
        request, so datetime.datetime.now() is the end moment of the request

        Parameters
        ----------
        request : urllib.request.Request
            The request object relevant to this call
        request_start : datetime.datetime
            The moment the request was handed off to urllib.request.urlopen()
        response_start : datetime.datetime
            The moment there were bytes from the server to process
        response_size : int
            The size of the response from the server in bytes
        chunk_size : int
            The number of cutout files recieved in this request
        """

        now = datetime.datetime.now()

        self._stat_accumulate("request_duration", response_start - request_start)
        self._stat_accumulate("response_duration", now - response_start)
        self._stat_accumulate("request_size_bytes", len(request.data))
        self._stat_accumulate("response_size_bytes", response_size)
        self._stat_accumulate("snapshots", chunk_size)

        self._print_stats(logging.DEBUG)


class FailedChunkCollector:
    """Collection system for chunks of sky locations where the request for a chunk of cutouts failed.

    Keeps track of all variable_fields plus object_id for failed chunks

    save() dumps these chunks using astropy.table.Table.write()

    """

    def __init__(self, filepath: Path, **kwargs):
        """_summary_

        Parameters
        ----------
        filepath : Path
            File to read in if we are resuming a download, and where to save the failed chunks after.
            If the file does not exist yet an empty state is initialized.

        **kwargs : dict
            Keyword args passed to astropy.table.Table.read() and write() in the case that a file is used.
            Should only be used to control file format, not read/write semantics
        """
        self.__dict__.update({key: [] for key in variable_fields + ["object_id"]})
        self.seen_object_ids = set()
        self.filepath = filepath.resolve()
        self.format_kwargs = kwargs

        # If there is a failed chunk file from a previous run,
        # Read it in to initialize us
        if filepath.exists():
            prev_failed_chunks = Table.read(filepath)
            for key in variable_fields + ["object_id"]:
                column_as_list = prev_failed_chunks[key].data.tolist()
                self.__dict__[key] += column_as_list
            logger.debug(f"Adding object ID :{self.object_id} to failed list")

            self.seen_object_ids = {id for id in self.object_id}

        self.count = len(self.seen_object_ids)
        logger.debug(f"Failed chunk handler initialized with {self.count} objects")

    def __enter__(self):
        return self.hook

    def __exit__(self, exc_type, exc_value, traceback):
        self.save()

    def hook(self, rects: list[dC.Rect], exception: Exception, attempts: int):
        """Called when dc.Download fails to download a chunk of rects

        Parameters
        ----------
        rects : list[dC.Rect]
            The list of rect objects that were requested from the server
        exception : Exception
            The exception that was thrown on the final attempt to request this chunk
        attempts : int
            The number of attempts that were made to request this chunk

        """

        for rect in rects:
            # Relies on the name format set up in create_rects to work properly
            object_id = int(rect.name.split("_")[0])

            if object_id not in self.seen_object_ids:
                self.seen_object_ids.add(object_id)

                self.object_id.append(object_id)

                for key in variable_fields:
                    self.__dict__[key].append(rect.__dict__[key])

                self.count += 1
        logger.debug(f"Failed chunk handler processed {len(rects)} rects and is now of size {self.count}")

    def save(self):
        """
        Saves the current set of failed locations to the path specified.
        If no failed locations were saved by the hook, this function does nothing.
        """
        if self.count == 0:
            return
        else:
            # convert our class-member-based representation to an astropy table.
            for key in variable_fields + ["object_id"]:
                self.__dict__[key] = np.array(self.__dict__[key])

            missed = Table({key: self.__dict__[key] for key in variable_fields + ["object_id"]})

            # note that the choice to do overwrite=True here and to read in the entire fits file in
            # ___init__() is necessary because snapshots corresponding to the same object may cross
            # chunk boundaries decided by dC.download.
            #
            # Since we are de-duplicating rects by object_id, we need to read in all rects from a prior
            # run, and we therefore replace the file we were passed.
            missed.write(self.filepath, overwrite=True, **self.format_kwargs)


def download_cutout_group(rects: list[dC.Rect], cutout_dir: Union[str, Path], user, password, **kwargs):
    """Download cutouts to the given directory

    Calls downloadCutout.download, so supports long lists of rects beyond the limits of the HSC web API

    Parameters
    ----------
    rects : list[dC.Rect]
        The rects we would like to download
    cutout_dir : Union[str, Path]
        The directory to put the files
    user : string
        Username for HSC's download service to use
    password : string
        Password for HSC's download service to use
    **kwargs: dict
        Additonal arguments for downloadCutout.download. See downloadCutout.download for details
    """

    with working_directory(Path(cutout_dir)):
        with (
            DownloadStats() as stats_hook,
            FailedChunkCollector(Path("failed_locations.fits"), format="fits") as failed_chunk_hook,
        ):
            dC.download(
                rects,
                user=user,
                password=password,
                onmemory=False,
                request_hook=stats_hook,
                failed_chunk_hook=failed_chunk_hook,
                resume=True,
                **kwargs,
            )

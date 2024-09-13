import itertools
import logging
from pathlib import Path
from threading import Thread
from typing import Optional

from astropy.table import Table, hstack

import fibad.downloadCutout.downloadCutout as dC
from fibad.download_stats import DownloadStats

logger = logging.getLogger(__name__)


class Downloader:
    """Class with primarily static methods to namespace downloader related constants and functions."""

    # These are the fields that are allowed to vary across the locations
    # input from the catalog fits file. Other values for HSC cutout server
    # must be provided by config.
    VARIABLE_FIELDS = ["tract", "ra", "dec"]

    # These are the column names we retain when writing a rect out to the manifest.fits file
    RECT_COLUMN_NAMES = VARIABLE_FIELDS + ["filter", "sw", "sh", "rerun", "type"]

    MANIFEST_FILE_NAME = "manifest.fits"

    @staticmethod
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

        fits_file = Path(config.get("fits_file", "")).resolve()
        logger.info(f"Reading in fits catalog: {fits_file}")
        # Filter the fits file for the fields we want
        column_names = ["object_id"] + Downloader.VARIABLE_FIELDS
        locations = Downloader.filterfits(fits_file, column_names)

        # If offet/length specified, filter to that length
        offset = config.get("offset", 0)
        end = offset + config.get("num_sources", None)
        if end is not None:
            locations = locations[offset:end]

        cutout_path = Path(config.get("cutout_dir")).resolve()
        logger.info(f"Downloading cutouts to {cutout_path}")

        logger.info("Making a list of cutouts...")
        # Make a list of rects to pass to downloadCutout
        rects = Downloader.create_rects(
            locations, offset=0, default=Downloader.rect_from_config(config), path=cutout_path
        )

        logger.info("Checking the list against currently downloaded cutouts...")
        # Prune any previously downloaded rects from our list using the manifest from the previous download
        rects = Downloader._prune_downloaded_rects(cutout_path, rects)

        # Early return if there is nothing to download.
        if len(rects) == 0:
            logger.info("Download already complete according to manifest.")
            return

        # Create thread objects for each of our worker threads
        num_threads = config.get("concurrent_connections", 10)

        if num_threads > 10:
            RuntimeError("More than 10 concurrent connections to HSC is disallowed on a per-user basis")

        # If we are using more than one connection, cut the list of rectangles into
        # batches, one batch for each thread.
        # TODO: Remove this in favor of itertools.batched() when we no longer support python < 3.12.
        def _batched(iterable, n):
            """Brazenly copied and pasted from the python 3.12 documentation.
            This is a dodgy version of a new itertools function in Python 3.12 called itertools.batched()
            """
            if n < 1:
                raise ValueError("n must be at least one")
            iterator = iter(iterable)
            while batch := tuple(itertools.islice(iterator, n)):
                yield batch

        logger.info("Dividing cutouts among threads...")
        thread_rects = list(_batched(rects, int(len(rects) / num_threads))) if num_threads != 1 else [rects]

        # Empty dictionaries for the threads to create download manifests in
        thread_manifests = [dict() for _ in range(num_threads)]

        shared_thread_args = (
            config["username"],
            config["password"],
            DownloadStats(print_interval_s=config.get("stats_print_interval", 60)),
        )

        shared_thread_kwargs = {
            "retrywait": config.get("retry_wait", 30),
            "retries": config.get("retries", 3),
            "timeout": config.get("timeout", 3600),
            "chunksize": config.get("chunk_size", 990),
        }

        download_threads = [
            Thread(
                target=Downloader.download_thread,
                name=f"thread_{i}",
                daemon=True,  # daemon so these threads will die when the main thread is interrupted
                args=(thread_rects[i],)  # rects
                + shared_thread_args  # username, password, download stats
                + (i, thread_manifests[i]),  # thread_num, manifest
                kwargs=shared_thread_kwargs,
            )
            for i in range(num_threads)
        ]

        try:
            logger.info(f"Started {len(download_threads)} request threads")
            [thread.start() for thread in download_threads]
            [thread.join() for thread in download_threads]
        finally:  # Ensure manifest is written even when we get a KeyboardInterrupt during download
            Downloader.write_manifest(thread_manifests, cutout_path)

        logger.info("Done")

    @staticmethod
    def _prune_downloaded_rects(cutout_path: Path, rects: list[dC.Rect]) -> list[dC.Rect]:
        """Prunes already downloaded rects using the manifest in `cutout_path`. `rects` passed in is
        mutated by this operation

        Parameters
        ----------
        cutout_path : Path
            Where on the filesystem to find the manifest
        rects : list[dC.Rect]
            List of rects from which we want to prune previously downloaded rects

        Returns
        -------
        list[dC.Rect]
            Returns `rects` that was passed in. This is only to enable explicit style at the call site.
            ` rects` is mutated by this function.

        Raises
        ------
        RuntimeError
            When there is an issue reading the manifest file, or the manifest file corresponds to a different
            set of cutouts than the current download being attempted
        """
        # print(rects)
        # Read in any prior manifest.
        prior_manifest = Downloader.read_manifest(cutout_path)

        # If we found a manifest, we are resuming a download
        if len(prior_manifest) != 0:
            # Filter rects to figure out which ones are completely downloaded.
            # This operation consumes prior_manifest in the process
            rects[:] = [rect for rect in rects if Downloader._keep_rect(rect, prior_manifest)]

            # if prior_manifest was not completely consumed, than the earlier download attempted
            # some sky locations which would not be included in the current download, and we have
            # a problem.
            if len(prior_manifest) != 0:
                # print(len(prior_manifest))
                # print (prior_manifest)
                raise RuntimeError(
                    f"""{cutout_path/Downloader.MANIFEST_FILE_NAME} describes a download with
sky locations that would not be downloaded in the download currently being attempted. Are you sure you are
resuming the correct download? Deleting the manifest and cutout files will start the download from scratch"""
                )

        return rects

    @staticmethod
    def _keep_rect(location_rect: dC.Rect, prior_manifest: dict[dC.Rect, str]) -> bool:
        """Private helper function to prune_downloaded_rects which operates the inner loop
        of the prune function, and allows it to be written as a list comprehension.

        This function decides element-by-element for our rects that we want to download whether
        or not these rects have already been downloaded in a prior download, given the manifest
        from that prior download.

        Parameters
        ----------
        location_rect : dC.Rect
            A rectangle on the sky that we are considering downloading.

        prior_manifest : dict[dC.Rect,str]
            The manifest of the prior download. This object is slowly consumed by repeated calls
            to this function. When the return value is False, all manifest entries corresponding to the
            passed in location_rect have been removed.

        Returns
        -------
        bool
            Whether this sky location `location_rect` should be included in the download
        """
        # Keep any location rect if the manifest passed has nothing in it.
        if len(prior_manifest) == 0:
            return True

        keep_rect = False
        for filter_rect in location_rect.explode():
            # Consume any matching manifest entry, keep the rect if
            # 1) The manifest entry doesn't exist -> pop returns None
            # 2) The manifest entry contains "Attempted" for the filename -> The corresponding file wasn't
            #    successfully downloaded
            matching_manifest_entry = prior_manifest.pop(filter_rect, None)
            if matching_manifest_entry is None or matching_manifest_entry == "Attempted":
                keep_rect = True

        return keep_rect

    @staticmethod
    def write_manifest(thread_manifests: list[dict[dC.Rect, str]], file_path: Path):
        """Write out manifest fits file that is an inventory of the download.
        The manifest fits file should have columns object_id, ra, dec, tract, filter, filename

        If filename is empty string ("") that means a download attempt was made, but did not succeed
        If the object is not present in the manifest, no download was attempted.
        If the object is present in the manifest and the filename is not empty string that file exists
        and downloaded successfully.

        This file respects the existence of other manifest files in the directory and operates additively.
        If a manifest file is present from an earlier download, this function will read that manifest in,
        and include the entire content of that manifest in addition to the manifests passed in.

        The format of the manifest file has the following columns

        object_id: The object ID from the original catalog
        filename: The file name where the file can be found OR the string "Attempted" indicating the download
                  did not complete successfully.
        tract: The HSC tract ID number this either comes from the catalog or is the tract ID returned by the
               cutout server for downloaded files.

        ra: Right ascension in degrees of the center of the cutout box
        dec: Declination in degrees of the center of the cutout box
        filter: The name of the filter requested
        sw: Semi-width of the cutout box in degrees
        sh: Semi-height of the cutout box in degrees
        rerun: The data release in use e.g. pdr3_wide
        type: coadd, warp, or other values allowed by the HSC docs

        Parameters
        ----------
        thread_manifests : list[dict[dC.Rect,str]]
            Manifests mapping rects -> Filename or status message. Each manifest came from a separate thread.

        file_path : Path
            Full path to the location where the manifest file ought be written. The manifest file will be
            named manifest.fits
        """
        logger.info("Assembling download manifest")
        # Start building a combined manifest from all threads from the ground truth of the prior manifest
        # in this directory, which we will be overwriting.
        combined_manifest = Downloader.read_manifest(file_path)

        # Combine all thread manifests with the prior manifest, so that the current status of a downloaded
        # rect overwrites any status from the prior run (which is no longer relevant.)
        for manifest in thread_manifests:
            combined_manifest.update(manifest)

        logger.info(f"Writing out download manifest with {len(combined_manifest)} entries.")

        # Convert the combined manifest into an astropy table by building a dict of {column_name: column_data}
        # for all the fields in a rect, plus our object_id and filename.
        column_names = Downloader.RECT_COLUMN_NAMES + ["filename", "object_id"]
        columns = {column_name: [] for column_name in column_names}

        for rect, msg in combined_manifest.items():
            # This parsing relies on the name format set up in create_rects to work properly
            # We parse the object_id from rect.name in case the filename is "Attempted" because the
            # download did not finish.
            rect_filename = Path(rect.name).name
            object_id = int(rect_filename.split("_")[0])
            columns["object_id"].append(object_id)

            # Remove the leading path from the filename if any.
            filename = Path(msg).name
            columns["filename"].append(filename)

            for key in Downloader.RECT_COLUMN_NAMES:
                columns[key].append(rect.__dict__[key])

        # print(columns)
        # for key, val in columns.items():
        #    print (key, len(val), val)

        manifest_table = Table(columns)
        manifest_table.write(file_path / Downloader.MANIFEST_FILE_NAME, overwrite=True, format="fits")

        logger.info("Finished writing download manifest")

    @staticmethod
    def read_manifest(file_path: Path) -> dict[dC.Rect, str]:
        """Read the manifest.fits file from the given directory and return its contents as a dictionary with
        downloadCutout.Rectangles as keys and filenames as values.

        If now manifest file is found, an empty dict is returned.

        Parameters
        ----------
        file_path : Path
            Where to find the manifest file

        Returns
        -------
        dict[dC.Rect, str]
            A dictionary containing all the rects in the manifest and all the filenames, or empty dict if no
            manifest is found.
        """
        filename = file_path / Downloader.MANIFEST_FILE_NAME
        if filename.exists():
            manifest_table = Table.read(filename, format="fits")
            rects = Downloader.create_rects(
                locations=manifest_table, fields=Downloader.RECT_COLUMN_NAMES, path=file_path
            )
            return {rect: filename for rect, filename in zip(rects, manifest_table["filename"])}
        else:
            return {}

    @staticmethod
    def download_thread(
        rects: list[dC.Rect],
        user: str,
        password: str,
        stats: DownloadStats,
        thread_num: int,
        manifest: dict[dC.Rect, str],
        **kwargs,
    ):
        """Download cutouts to the given directory. Called in its own thread with an id number.

        Calls downloadCutout.download, so supports long lists of rects beyond the limits of the HSC web API

        Parameters
        ----------
        rects : list[dC.Rect]
            The rects we would like to download
        user : string
            Username for HSC's download service to use
        password : string
            Password for HSC's download service to use
        stats : DownloadStats
            Instance of DownloadStats to use for stats tracking.
        thread_num : int,
            The ID number of thread we are, sequential from zero to num_threads-1
        manifest:
            A dictionary from dC.Rect to filename which we will fill in in as we download rects. This is the
            chief returned piece of data from each thread.
        **kwargs: dict
            Additonal arguments for downloadCutout.download. See downloadCutout.download for details
        """
        logger.info(f"Thread {thread_num} starting download of {len(rects)} rects")
        with stats as stats_hook:
            dC.download(
                rects,
                user=user,
                password=password,
                onmemory=False,
                request_hook=stats_hook,
                manifest=manifest,
                **kwargs,
            )

    # TODO add error checking
    @staticmethod
    def filterfits(filename: Path, column_names: list[str]) -> Table:
        """Read a fits file with the required column names for making cutouts

        The easiest way to make such a fits file is to select from the main HSC catalog

        Parameters
        ----------
        filename : str
            The fits file to read in
        column_names : list[str]
            The columns that are selected from the file and returned in the astropy Table.

        Returns
        -------
        Table
        Returns an astropy table containing only the fields specified in column_names
        """
        t = Table.read(filename)
        columns = [t[column] for column in column_names]
        return hstack(columns, uniq_col_name="{table_name}", table_names=column_names)

    @staticmethod
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
            image=config.get("image"),
            mask=config.get("mask"),
            variance=config.get("variance"),
        )

    @staticmethod
    def create_rects(
        locations: Table,
        path: Path,
        offset: int = 0,
        default: dC.Rect = None,
        fields: Optional[list[str]] = None,
    ) -> list[dC.Rect]:
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
        path : Path
            Directory where the cutuout files ought live. Used to generate file names on the rect object.
        offset : int, optional
            Index to start the `lineno` field in the rects at, by default 0. The purpose of this is to allow
            multiple downloads on different sections of a larger source list without file name clobbering
            during the download phase. This is important because `lineno` in a rect can becomes a file name
            parameter The offset is intended to be the index of the start of the locations table within some
            larger fits file.
        default : dC.Rect, optional
            The default Rect that contains properties common to all sky locations, by default None

        fields : list[str], optional
            Default fields to pull from the locations table. If not provided, defaults to
            ["tract", "ra", "dec"]

        Returns
        -------
        list[dC.Rect]
            Rects populated with sky locations from the table
        """
        rects = []
        fields = fields if fields else Downloader.VARIABLE_FIELDS
        for index, location in enumerate(locations):
            args = {field: location[field] for field in fields}
            args["lineno"] = index + offset

            # tracts are ints in the fits files and dC.rect constructor wants them as str
            args["tract"] = str(args["tract"])

            # Sets the file name on the rect to be the object_id, also includes other rect fields
            # which are interpolated at save time, and are native fields of dc.Rect.
            args["name"] = str(
                path / f"{location['object_id']}_{{type}}_{{ra:.5f}}_{{dec:+.5f}}_{{tract}}_{{filter}}"
            )
            rect = dC.Rect.create(default=default, **args)
            rects.append(rect)

        # We sort rects here so they end up tract,ra,dec ordered across all requests made in all threads
        # Threads do their own sorting prior to all chunked request in downloadCutout.py; however
        # sorting at this stage will allow a greater number of rects that are co-located in the sky
        # to end up in the same thread and therefore be sorted into the same request.
        rects.sort()

        return rects

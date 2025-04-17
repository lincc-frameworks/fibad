import functools

import astropy.units as u
from astropy.table import Table
from torch.utils.data import Dataset

from .data_set_registry import HyraxDataset


class LSSTDataset(HyraxDataset, Dataset):
    """LSSTDataset: A dataset to access deep_coadd images from lsst pipelines
    via the butler. Must be run in an RSP.

    Input catalog is hardcoded
    Cutout code incorrect near the edges of a tract or patch

    Batteries not included, use at your own risk etc.
    """

    # Hardcode catalog for now (will need to get from a file via config)
    INPUT_CATALOG = Table(
        {
            "ra": [53.182137366954045 * u.deg],
            "dec": [-28.27055798839844 * u.deg],
            "sh": [(20 * u.arcsec).to(u.deg)],
            "sw": [(20 * u.arcsec).to(u.deg)],
        }
    )

    # Hardcode butler for now (will need to get from config or environment)
    BUTLER_CONFIG = {
        "repo": "/repo/main",
        # collections = 'LSSTComCam/runs/DRP/DP1/w_2025_07/DM-48940'
        # collections = 'LSSTComCam/runs/DRP/DP1/w_2025_09/DM-49235'
        # collections = 'LSSTComCam/runs/DRP/DP1/w_2025_10/DM-49359'
        # collections = 'LSSTComCam/runs/DRP/DP1/w_2025_11/DM-49472'
        #'collection': 'LSSTComCam/runs/DRP/DP1/v29_0_0_rc2/DM-49592',
        "collection": "LSSTComCam/runs/DRP/DP1/v29_0_0_rc6/DM-50098",
        "skymap": "lsst_cells_v1",
    }

    BANDS = ["g", "r", "i"]

    def __init__(self, config):
        try:
            import lsst.daf.butler as butler
        except ImportError as e:
            msg = "LSSTDataset can only be used in a Rubin Software Platform environment"
            msg += " with access to the lsst pipeline tools."
            raise ImportError(msg) from e

        super().__init__(config)
        self.butler = butler.Butler(
            LSSTDataset.BUTLER_CONFIG["repo"], collections=LSSTDataset.BUTLER_CONFIG["collection"]
        )

        self.skymap = self.butler.get("skyMap", {"skymap": LSSTDataset.BUTLER_CONFIG["skymap"]})

    def __len__(self):
        return len(LSSTDataset.INPUT_CATALOG)

    def __getitem__(self, idxs):
        rows = LSSTDataset.INPUT_CATALOG[idxs]
        rows = rows if isinstance(rows, list) else [rows]
        cutouts = [self._fetch_single_cutout(row) for row in rows]

        return cutouts if len(cutouts) > 1 else cutouts[0]

    # def __getitems__(self, idxs):
    #     return __getitem__(self, idxs)

    def _parse_box(self, patch, row):
        """
        Return a Box2I representing the desired cutout in pixel space, given a "row" of catalog data
        which includes the semi-height (sh) and semi-width (sw) in degrees desired for the cutout.
        """
        from lsst.geom import Box2D, Box2I, degrees

        radec = self._parse_sphere_point(row)
        sw = row["sw"] * degrees
        sh = row["sh"] * degrees

        # Ra/Dec is left handed on the sky. Pixel coordinates are right handed on the sky.
        # In the variable names below min/max mean the min/max coordinate values in the
        # right-handed pixel space

        # Move + in ra (0.0) for width and - in dec (270.0) along a great circle
        min_pt_sky = radec.offset(0.0 * degrees, sw).offset(270.0 * degrees, sh)
        # Move - in ra (180.0) for width and + in dec (90.0) along a great circle
        max_pt_sky = radec.offset(180.0 * degrees, sw).offset(90.0 * degrees, sh)

        wcs = patch.getWcs()
        min_pt_pixel_f = wcs.skyToPixel(min_pt_sky)
        max_pt_pixel_f = wcs.skyToPixel(max_pt_sky)
        box_f = Box2D(min_pt_pixel_f, max_pt_pixel_f)
        return Box2I(box_f, Box2I.EXPAND)

    def _parse_sphere_point(self, row):
        """
        Return a SpherePoint with the ra and deck given in the "row" of catalog data.
        Row must include the RA and dec as "ra" and "dec" columns respectively
        """
        from lsst.geom import SpherePoint, degrees

        ra = row["ra"]
        dec = row["dec"]
        return SpherePoint(ra, dec, degrees)

    def _get_tract_patch(self, row):
        """
        Return (tractInfo, patchInfo) for a given row.

        This function only returns the single principle tract and patch in the case of overlap.
        """
        radec = self._parse_sphere_point(row)
        tract_info = self.skymap.findTract(radec)

        # Note: How to look up all tracts for the ra/dec (this will happen sometimes because of
        # overlapping tracts)
        #
        # [(tract, tract.findPatch(radec)) for tract in skymap.findAllTracts(radec)]
        # print(tractInfo.getId(), patchInfo.sequential_index)
        #
        return (tract_info, tract_info.findPatch(radec))

    # super basic patch caching
    @functools.lru_cache(maxsize=128)  # noqa: B019
    def _request_patch(self, tract_index, patch_index):
        """
        Request a patch from the butler. This will be a list of
        lsst.afw.image objects each corresponding to the configured
        bands

        Uses functools.lru_cache for basic in-memory caching.
        """
        data = []

        # Get the patch images we need
        for band in LSSTDataset.BANDS:
            # Set up the data dict
            butler_dict = {
                "tract": tract_index,
                "patch": patch_index,
                "skymap": LSSTDataset.BUTLER_CONFIG["skymap"],
                "band": band,
            }

            # pull from butler
            image = self.butler.get("deep_coadd", butler_dict)
            data.append(image.getImage())
        return data

    def _fetch_single_cutout(self, row):
        """
        Make a single cutout, returning a torch tensor.

        Does not handle edge-of-tract/patch type edge cases, will only work near
        center of a patch.
        """
        import numpy as np
        from torch import from_numpy

        tract_info, patch_info = self._get_tract_patch(row)
        box_i = self._parse_box(patch_info, row)

        patch_images = self._request_patch(tract_info.getId(), patch_info.sequential_index)

        # Actually perform a cutout
        data = [image[box_i].getArray() for image in patch_images]

        # Convert to torch format
        data_np = np.array(data)
        data_torch = from_numpy(data_np.astype(np.float32))

        return data_torch

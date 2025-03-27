import logging

from hyrax.pytorch_ignite import setup_dataset

logger = logging.getLogger(__name__)


def run(config):
    """Rebuild a broken download manifest

    Parameters
    ----------
    config : dict
        The parsed config file as a nested
        dict
    """
    from .data_sets.hsc_data_set import HSCDataSet

    config["rebuild_manifest"] = True

    data_set = setup_dataset(config)

    if not isinstance(data_set, HSCDataSet):
        msg = "Invalid to run rebuild manafest except on an HSCDataSet."
        raise RuntimeError(msg)

    logger.info("Starting rebuild of manifest")

    data_set._rebuild_manifest(config)

    logger.info("Finished Rebuild Manifest")

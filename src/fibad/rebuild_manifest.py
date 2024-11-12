import logging

from fibad.pytorch_ignite import setup_model_and_dataset

logger = logging.getLogger(__name__)


def run(config):
    """Rebuild a broken download manifest

    Parameters
    ----------
    config : dict
        The parsed config file as a nested
        dict
    """

    _, data_set = setup_model_and_dataset(config, split=config["train"]["split"])

    logger.info("Starting rebuild of manifest")

    data_set.rebuild_manifest(config)

    logger.info("Finished Rebuild Manifest")

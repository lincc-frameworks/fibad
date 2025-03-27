import logging

from hyrax.pytorch_ignite import setup_dataset

logger = logging.getLogger(__name__)


def run(config):
    """Prepare the dataset for a given model and data loader.

    Parameters
    ----------
    config : dict
        The parsed config file as a nested
        dict
    """

    data_set = setup_dataset(config)

    logger.info("Finished Prepare")
    return data_set

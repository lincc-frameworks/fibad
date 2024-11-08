import logging

from fibad.pytorch_ignite import setup_model_and_dataset

logger = logging.getLogger(__name__)


def run(config):
    """Prepare the dataset for a given model and data loader.

    Parameters
    ----------
    config : dict
        The parsed config file as a nested
        dict
    """

    _, data_set = setup_model_and_dataset(config, split=config["train"]["split"])

    logger.info("Finished Prepare")
    return data_set

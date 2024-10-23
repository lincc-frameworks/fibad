import logging

from fibad.config_utils import create_results_dir, log_runtime_config
from fibad.pytorch_ignite import create_trainer, dist_data_loader, setup_model_and_dataset

logger = logging.getLogger(__name__)


def run(config):
    """Run the training process for a given model and data loader.

    Parameters
    ----------
    config : dict
        The parsed config file as a nested
        dict
    """
    # Create a results directory
    results_dir = create_results_dir(config, "train")
    log_runtime_config(config, results_dir)

    model, data_set = setup_model_and_dataset(config, split=config["train"]["split"])
    data_loader = dist_data_loader(data_set, config)

    # Create trainer, a pytorch-ignite `Engine` object
    trainer = create_trainer(model, config, results_dir)

    # Run the training process
    trainer.run(data_loader, max_epochs=config["train"]["epochs"])

    # Save the trained model
    model.save(results_dir / config["train"]["weights_filepath"])

    logger.info("Finished Training")

import logging

from fibad.config_utils import create_results_dir, log_runtime_config
from fibad.pytorch_ignite import create_trainer, setup_model_and_dataloader

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

    model, data_loader = setup_model_and_dataloader(config)

    # Create trainer, a pytorch-ignite `Engine` object
    trainer = create_trainer(model)

    # Run the training process
    trainer.run(data_loader, max_epochs=config["model"]["epochs"])

    # Save the trained model
    model.save(results_dir / config["model"]["weights_filepath"])

    logger.info("Finished Training")

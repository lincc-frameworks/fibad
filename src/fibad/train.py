import logging
from pathlib import Path

import mlflow
from tensorboardX import SummaryWriter

from fibad.config_utils import create_results_dir, log_runtime_config
from fibad.gpu_monitor import GpuMonitor
from fibad.pytorch_ignite import (
    create_trainer,
    create_validator,
    dist_data_loader,
    setup_dataset,
    setup_model,
)

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

    # Create a tensorboardX logger
    tensorboardx_logger = SummaryWriter(log_dir=results_dir)

    # Instantiate the model and dataset
    data_set = setup_dataset(config, split=config["train"]["split"])
    model = setup_model(config, data_set)

    # Create a data loader for the training set
    train_data_loader = dist_data_loader(data_set, config, "train")

    # Create validation_data_loader if a validation split is defined in data_set
    validation_data_loader = dist_data_loader(data_set, config, "validate")

    # Create trainer, a pytorch-ignite `Engine` object
    trainer = create_trainer(model, config, results_dir, tensorboardx_logger)

    # Create a validator if a validation data loader is available
    if validation_data_loader is not None:
        create_validator(model, config, results_dir, tensorboardx_logger, validation_data_loader, trainer)

    monitor = GpuMonitor(tensorboard_logger=tensorboardx_logger)

    results_root_dir = Path(config["general"]["results_dir"]).resolve()
    mlflow.set_tracking_uri("file://" + str(results_root_dir / "mlflow"))
    mlflow.set_experiment("notebook")

    with mlflow.start_run(log_system_metrics=True):
        mlflow.log_params(config["model"])
        mlflow.log_param("epochs", config["train"]["epochs"])
        mlflow.log_param("batch_size", config["data_loader"]["batch_size"])

        # Run the training process
        trainer.run(train_data_loader, max_epochs=config["train"]["epochs"])

        mlflow.pytorch.log_model(model, "models")

    # Save the trained model
    model.save(results_dir / config["train"]["weights_filepath"])
    monitor.stop()

    logger.info("Finished Training")
    tensorboardx_logger.close()

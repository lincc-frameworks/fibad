import logging

#! We should not import anything that is pytorch or pytorch-ignite specific
#! in this file. It should be kept as ML framework agnostic as possible.
from ignite.engine import Events
from ignite.handlers import global_step_from_engine
from ignite.handlers.tensorboard_logger import TensorboardLogger

from fibad.config_utils import create_results_dir, log_runtime_config
from fibad.pytorch_ignite import create_trainer, create_validator, dist_data_loader, setup_model_and_dataset

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

    #! TODO: dist_data_loader will check to see if the data_set will support a
    #! train and validate split, but the rest of this code doesn't perform those
    #! checks. We shouldn't create the `validator` engine if the data_set doesn't
    #! support a validation split.
    train_data_loader = dist_data_loader(data_set, config, "train")
    validation_data_loader = dist_data_loader(data_set, config, "validate")

    # Create trainer, a pytorch-ignite `Engine` object
    trainer = create_trainer(model, config, results_dir)
    validator = create_validator(model, config, results_dir)

    #! We should move all of the `@trainer.on` and `@validator.on` decorators
    #! out of this method to some thing else that will allow us to keep this
    #! function ML framework agnostic.
    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation():
        validator.run(validation_data_loader)

    #! Tensorboard logging is fine for now, but I think we should opt to use
    #! `tensorboardX` instead of `ignite`'s tensorboard logger. This will allow
    #! us to use the same logging system for all future ML frameworks.
    tensorboard_logger = TensorboardLogger(log_dir=results_dir)

    tensorboard_logger.attach_output_handler(
        validator,
        event_name=Events.EPOCH_COMPLETED,
        tag="training/validation",
        output_transform=lambda loss: loss,
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )

    tensorboard_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=10),
        tag="training/training",
        output_transform=lambda loss: loss,
        metric_names="all",
    )

    # Run the training process
    trainer.run(train_data_loader, max_epochs=config["train"]["epochs"])

    # Save the trained model
    model.save(results_dir / config["train"]["weights_filepath"])

    logger.info("Finished Training")
    tensorboard_logger.close()

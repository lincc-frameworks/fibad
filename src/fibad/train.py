import logging

import torch
from ignite import distributed as idist
from ignite.engine import Engine, Events

from fibad.data_loaders.data_loader_registry import fetch_data_loader_class
from fibad.models.model_registry import fetch_model_class

logger = logging.getLogger(__name__)


def run(config):
    """Run the training process for a given model and data loader.

    Parameters
    ----------
    config : dict
        The parsed config file as a nested
        dict
    """

    # Fetch data loader class specified in config and create an instance of it
    data_loader_cls = fetch_data_loader_class(config)
    data_loader = data_loader_cls(config.get("data_loader", {}))

    # Get the pytorch.dataset from dataloader, and use it to create a distributed dataloader
    data_set = data_loader.data_set()
    dist_data_loader = _train_data_loader(data_set, config.get("data_loader", {}))

    # Fetch model class specified in config and create an instance of it
    model_cls = fetch_model_class(config)
    model = model_cls(model_config=config.get("model", {}), shape=data_loader.shape())

    # Create trainer, a pytorch-ignite `Engine` object
    trainer = _create_trainer(model)

    # Run the training process
    trainer.run(dist_data_loader, max_epochs=config.get("model", {}).get("epochs", 2))

    # Save the trained model
    model.save()
    logger.info("Finished Training")


def _train_data_loader(data_set, config):
    # ~ idist.auto_dataloader will accept a **kwargs parameter, and pass values
    # ~ through to the underlying pytorch DataLoader.
    # ~ Currently, our config includes unexpected keys like `name`, that cause
    # ~ an exception. It would be nice to reduce this to:
    # ~ `data_loader = idist.auto_dataloader(data_set, **config)`
    data_loader = idist.auto_dataloader(
        data_set,
        batch_size=config.get("batch_size", 4),
        shuffle=config.get("shuffle", True),
        num_workers=config.get("num_workers", 2),
    )

    return data_loader


def _create_trainer(model):
    """This function is originally copied from here:
    https://github.com/pytorch-ignite/examples/blob/main/tutorials/intermediate/cifar10-distributed.py#L164

    It was substantially trimmed down to make it easier to understand.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.

    Returns
    -------
    pytorch-ignite.Engine
        Engine object that will be used to train the model.
    """
    # Get currently available device for training, and set the model to use it
    device = idist.device()
    # logger.info(f"Training on device: {device}")
    model = idist.auto_model(model)

    # Extract `train_step` from model, which can be wrapped after idist.auto_model(...)
    if type(model) == torch.nn.parallel.DistributedDataParallel:
        inner_train_step = model.module.train_step
    elif type(model) == torch.nn.parallel.DataParallel:
        inner_train_step = model.module.train_step
    else:
        inner_train_step = model.train_step

    # Wrap the `train_step` so that batch data is moved to the appropriate device
    def train_step(engine, batch):
        #! This feels brittle, it would be worth revisiting this.
        #  We assume that the batch data will generally have two forms.
        # 1) A torch.Tensor that represents N samples.
        # 2) A tuple (or list) of torch.Tensors, where the first tensor is the
        # data, and the second is labels.
        batch = batch.to(device) if isinstance(batch, torch.Tensor) else tuple(i.to(device) for i in batch)

        return inner_train_step(batch)

    # Create the ignite `Engine` object
    trainer = Engine(train_step)

    @trainer.on(Events.STARTED)
    def log_training_start(trainer):
        logger.info(f"Training model on device: {device}")
        logger.info(f"Total epochs: {trainer.state.max_epochs}")

    @trainer.on(Events.EPOCH_STARTED)
    def log_epoch_start(trainer):
        logger.debug(f"Starting epoch {trainer.state.epoch}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(trainer):
        logger.info(f"Epoch {trainer.state.epoch} run time: {trainer.state.times['EPOCH_COMPLETED']:.2f}[s]")
        logger.info(f"Epoch {trainer.state.epoch} metrics: {trainer.state.output}")

    @trainer.on(Events.COMPLETED)
    def log_total_time(trainer):
        logger.info(f"Total training time: {trainer.state.times['COMPLETED']:.2f}[s]")

    return trainer

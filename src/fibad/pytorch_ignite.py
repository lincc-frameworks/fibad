import functools
import logging

import ignite.distributed as idist
import torch
from ignite.engine import Engine, Events

from fibad.config_utils import ConfigDict
from fibad.data_loaders.data_loader_registry import fetch_data_loader_class
from fibad.models.model_registry import fetch_model_class

logger = logging.getLogger(__name__)


def setup_model_and_dataloader(config: ConfigDict) -> tuple:
    """
    Construct the data loader and the model according to configuration.

    Primarily exists so the train and predict actions do this the same way.

    Parameters
    ----------
    config : ConfigDict
       The entire runtime config

    Returns
    -------
    tuple
        (model object, data loader object)
    """
    # Fetch data loader class specified in config and create an instance of it
    data_loader_cls = fetch_data_loader_class(config)
    data_loader = data_loader_cls(config)

    # Fetch model class specified in config and create an instance of it
    model_cls = fetch_model_class(config)
    model = model_cls(config=config, shape=data_loader.shape())

    # Get the pytorch.dataset from dataloader, and use it to create a distributed dataloader
    data_set = data_loader.data_set()

    return model, _dist_data_loader(data_set, config)


def _dist_data_loader(data_set, config):
    # ~ idist.auto_dataloader will accept a **kwargs parameter, and pass values
    # ~ through to the underlying pytorch DataLoader.
    # ~ Currently, our config includes unexpected keys like `name`, that cause
    # ~ an exception. It would be nice to reduce this to:
    # ~ `data_loader = idist.auto_dataloader(data_set, **config)`
    data_loader = idist.auto_dataloader(
        data_set,
        batch_size=config["data_loader"]["batch_size"],
        shuffle=config["data_loader"]["shuffle"],
        num_workers=config["data_loader"]["num_workers"],
    )

    return data_loader


def _extract_inner_function(model, funcname):
    # Extract `train_step` or `forward` from model, which can be wrapped after idist.auto_model(...)
    if (
        type(model) == torch.nn.parallel.DistributedDataParallel
        or type(model) == torch.nn.parallel.DataParallel
    ):
        inner_step = getattr(model.module, funcname)
    else:
        inner_step = getattr(model, funcname)

    return inner_step


# This wraps a model-specific function (func) to move data to the appropriate device.
def _inner_loop(func, device, engine, batch):
    #! This feels brittle, it would be worth revisiting this.
    #  We assume that the batch data will generally have two forms.
    # 1) A torch.Tensor that represents N samples.
    # 2) A tuple (or list) of torch.Tensors, where the first tensor is the
    # data, and the second is labels.
    batch = batch.to(device) if isinstance(batch, torch.Tensor) else tuple(i.to(device) for i in batch)

    return func(batch)


def _create_engine_loop(funcname, device, model):
    inner_step = _extract_inner_function(model, funcname)
    inner_loop = functools.partial(_inner_loop, inner_step, device)
    return inner_loop


def create_evaluator(model):
    """Based on create_trainer. This creates a pytorch ignite evaluator object with appropriate event
    handlers

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate

    Returns
    -------
    pytorch-ignite.Engine
        Engine object which when run will evaluate the model.
    """
    device = idist.device()
    model = idist.auto_model(model)
    evaluator = Engine(_create_engine_loop("forward", device, model))

    @evaluator.on(Events.STARTED)
    def log_eval_start(evaluator):
        logger.info(f"Evaluating model on device: {device}")
        logger.info(f"Total epochs: {evaluator.state.max_epochs}")

    @evaluator.on(Events.COMPLETED)
    def log_total_time(evaluator):
        logger.info(f"Total evaluation time: {evaluator.state.times['COMPLETED']:.2f}[s]")

    return evaluator


def create_trainer(model):
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

    trainer = Engine(_create_engine_loop("train_step", device, model))

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

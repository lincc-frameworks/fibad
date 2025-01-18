import functools
import logging
from pathlib import Path
from typing import Any, Callable, Union

import ignite.distributed as idist
import torch
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from tensorboardX import SummaryWriter
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from fibad.config_utils import ConfigDict
from fibad.data_sets.data_set_registry import fetch_data_set_class
from fibad.models.model_registry import fetch_model_class

logger = logging.getLogger(__name__)


def setup_dataset(config: ConfigDict, split: Union[str, bool] = False) -> Dataset:
    """Create a dataset object based on the configuration.

    Parameters
    ----------
    config : ConfigDict
        The entire runtime configuration
    split : Union[str,bool], optional
        The name of the split that we want to use. If False, use the entire
        dataset, by default False

    Returns
    -------
    Dataset
        An instance of the dataset class specified in the configuration
    """

    # Fetch data loader class specified in config and create an instance of it
    data_set_cls = fetch_data_set_class(config)
    data_set = data_set_cls(config, split)  # type: ignore[call-arg]

    return data_set


def setup_model(config: ConfigDict, dataset: Dataset) -> torch.nn.Module:
    """Create a model object based on the configuration.

    Parameters
    ----------
    config : ConfigDict
        The entire runtime configuration
    dataset : Dataset
        Only used to determine the input shape of the data

    Returns
    -------
    torch.nn.Module
        An instance of the model class specified in the configuration
    """

    # Fetch model class specified in config and create an instance of it
    model_cls = fetch_model_class(config)
    model = model_cls(config=config, shape=dataset.shape())  # type: ignore[attr-defined]

    return model


def dist_data_loader(data_set: Dataset, config: ConfigDict, split: str):
    """Create a Pytorch Ignite distributed data loader

    Parameters
    ----------
    data_set : Dataset
        A Pytorch Dataset object
    config : ConfigDict
        Fibad runtime configuration
    split : str
        The name of the split we want to use from the data set.

    Returns
    -------
    Dataloader (or an ignite-wrapped equivalent)
        This is the distributed dataloader, formed by calling ignite.distributed.auto_dataloader
    """

    #! Here we are allowing `train_sampler` and `validation_sampler` to become magic words.
    #! It would be nice to find a way to do this that doesn't require external libraries to
    #! know about these.
    if hasattr(data_set, "train_sampler") and split == "train":
        sampler = data_set.train_sampler
    elif hasattr(data_set, "validation_sampler") and split == "validate":
        sampler = data_set.validation_sampler
    else:
        sampler = None

    return idist.auto_dataloader(data_set, sampler=sampler, **config["data_loader"])


def create_engine(funcname: str, device: torch.device, model: torch.nn.Module) -> Engine:
    """Unified creation of the pytorch engine object for either an evaluator or trainer.

    This function will automatically unwrap a distributed model to find the necessary function, and construct
    the necessary functions to transfer data to the device on every batch, so model code can be the same no
    matter where the model is being run.

    Parameters
    ----------
    funcname : str
        The function name on the model that we will call in the core of the engine loop, and be called once
        per batch
    device : torch.device
        The device the engine will run the model on
    model : torch.nn.Module
        The Model the engine will be using
    """

    # This wraps a model-specific function (func) to move data to the appropriate device.
    def _inner_loop(func, device, engine, batch):
        #! This feels brittle, it would be worth revisiting this.
        #  We assume that the batch data will generally have two forms.
        # 1) A torch.Tensor that represents N samples.
        # 2) A tuple (or list) of torch.Tensors, where the first tensor is the
        # data, and the second is labels.
        batch = batch.to(device) if isinstance(batch, torch.Tensor) else tuple(i.to(device) for i in batch)
        return func(batch)

    def _create_process_func(funcname, device, model):
        inner_step = extract_model_method(model, funcname)
        inner_loop = functools.partial(_inner_loop, inner_step, device)
        return inner_loop

    return Engine(_create_process_func(funcname, device, model))


def extract_model_method(model, method_name):
    """Extract a method from a model, which may be wrapped in a DistributedDataParallel
     or DataParallel object. For instance, method_name could be `train_step` or
    `forward`.

    Parameters
    ----------
    model : nn.Module, DistributedDataParallel, or DataParallel
        The model to extract the method from
    method_name : str
        Name of the method to extract

    Returns
    -------
    Callable
        The method extracted from the model
    """
    wrapped = type(model) == DistributedDataParallel or type(model) == DataParallel
    return getattr(model.module if wrapped else model, method_name)


def create_evaluator(model: torch.nn.Module, save_function: Callable[[torch.Tensor], Any]) -> Engine:
    """Creates an evaluator engine
    Primary purpose of this function is to attach the appropriate handlers to an evaluator engine

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate

    save_function : Callable[[torch.Tensor], Any]
        A function which will receive Engine.state.output at the end of each iteration. The intent
        is for the results of evaluation to be saved.

    Returns
    -------
    pytorch-ignite.Engine
        Engine object which when run will evaluate the model.
    """
    device = idist.device()
    model.eval()
    model = idist.auto_model(model)
    evaluator = create_engine("forward", device, model)

    @evaluator.on(Events.STARTED)
    def log_eval_start(evaluator):
        logger.info(f"Evaluating model on device: {device}")
        logger.info(f"Total epochs: {evaluator.state.max_epochs}")

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_iteration_complete(evaluator):
        save_function(evaluator.state.output)

    @evaluator.on(Events.COMPLETED)
    def log_total_time(evaluator):
        logger.info(f"Total evaluation time: {evaluator.state.times['COMPLETED']:.2f}[s]")

    return evaluator


#! There will likely be a significant amount of code duplication between the
#! `create_trainer` and `create_validator` functions. We should find a way to
#! refactor this code to reduce duplication.
def create_validator(
    model: torch.nn.Module,
    config: ConfigDict,
    results_directory: Path,
    tensorboardx_logger: SummaryWriter,
    validation_data_loader: DataLoader,
    trainer: Engine,
) -> Engine:
    """This function creates a Pytorch Ignite engine object that will be used to
    validate the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train
    config : ConfigDict
        Fibad runtime configuration
    results_directory : Path
        The directory where training results will be saved
    tensorboardx_logger : SummaryWriter
        The tensorboard logger object
    validation_data_loader : DataLoader
        The data loader for the validation data
    trainer : Engine
        The engine object that will be used to train the model. We will use specific
        hooks in the trainer to determine when to run the validation engine.

    Returns
    -------
    pytorch-ignite.Engine
        Engine object that will be used to train the model.
    """

    device = idist.device()
    model = idist.auto_model(model)

    validator = create_engine("train_step", device, model)

    @validator.on(Events.STARTED)
    def set_model_to_eval_mode():
        model.eval()

    @validator.on(Events.COMPLETED)
    def set_model_to_train_mode():
        model.train()

    @validator.on(Events.EPOCH_COMPLETED)
    def log_training_loss():
        logger.info(f"Validation run time: {validator.state.times['EPOCH_COMPLETED']:.2f}[s]")
        logger.info(f"Validation metrics: {validator.state.output}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation():
        validator.run(validation_data_loader)

    def log_validation_loss(validator, trainer):
        step = trainer.state.get_event_attrib_value(Events.EPOCH_COMPLETED)
        tensorboardx_logger.add_scalar("training/validation/loss", validator.state.output["loss"], step)

    validator.add_event_handler(Events.EPOCH_COMPLETED, log_validation_loss, trainer)

    return validator


def create_trainer(
    model: torch.nn.Module, config: ConfigDict, results_directory: Path, tensorboardx_logger: SummaryWriter
) -> Engine:
    """This function is originally copied from here:
    https://github.com/pytorch-ignite/examples/blob/main/tutorials/intermediate/cifar10-distributed.py#L164

    It was substantially trimmed down to make it easier to understand.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train
    config : ConfigDict
        Fibad runtime configuration
    results_directory : Path
        The directory where training results will be saved
    tensorboardx_logger : SummaryWriter
        The tensorboard logger object

    Returns
    -------
    pytorch-ignite.Engine
        Engine object that will be used to train the model.
    """
    device = idist.device()
    model.train()
    model = idist.auto_model(model)
    trainer = create_engine("train_step", device, model)

    optimizer = extract_model_method(model, "optimizer")

    to_save = {
        "model": model,
        "optimizer": optimizer,
        "trainer": trainer,
    }

    #! We may want to move the checkpointing logic over to the `validator`.
    #! It was created here initially because this was the only place where the
    #! model training was happening.
    latest_checkpoint = Checkpoint(
        to_save,
        DiskSaver(results_directory, require_empty=False),
        n_saved=1,
        global_step_transform=global_step_from_engine(trainer),
        filename_pattern="{name}_epoch_{global_step}.{ext}",
    )

    def neg_loss_score(engine):
        return -engine.state.output["loss"]

    best_checkpoint = Checkpoint(
        to_save,
        DiskSaver(results_directory, require_empty=False),
        n_saved=1,
        global_step_transform=global_step_from_engine(trainer),
        score_name="loss",
        score_function=neg_loss_score,
        greater_or_equal=True,
    )

    if config["train"]["resume"]:
        prev_checkpoint = torch.load(config["train"]["resume"], map_location=device)
        Checkpoint.load_objects(to_load=to_save, checkpoint=prev_checkpoint)

    # results_root_dir = Path(config["general"]["results_dir"]).resolve()
    # mlflow_logger = MLflowLogger("file://" + str(results_root_dir / "mlflow"))

    @trainer.on(Events.STARTED)
    def log_training_start(trainer):
        logger.info(f"Training model on device: {device}")
        logger.info(f"Total epochs: {trainer.state.max_epochs}")

    @trainer.on(Events.EPOCH_STARTED)
    def log_epoch_start(trainer):
        logger.debug(f"Starting epoch {trainer.state.epoch}")

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_training_loss_tensorboard(trainer):
        step = trainer.state.get_event_attrib_value(Events.ITERATION_COMPLETED)
        tensorboardx_logger.add_scalar("training/training/loss", trainer.state.output["loss"], step)

    # mlflow_logger.attach_output_handler(
    #     trainer,
    #     event_name=Events.ITERATION_COMPLETED,
    #     tag="training",
    #     output_transform=lambda loss: {"loss": loss},
    # )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(trainer):
        logger.info(f"Epoch {trainer.state.epoch} run time: {trainer.state.times['EPOCH_COMPLETED']:.2f}[s]")
        logger.info(f"Epoch {trainer.state.epoch} metrics: {trainer.state.output}")

    trainer.add_event_handler(Events.EPOCH_COMPLETED, latest_checkpoint)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, best_checkpoint)

    @trainer.on(Events.COMPLETED)
    def log_total_time(trainer):
        logger.info(f"Total training time: {trainer.state.times['COMPLETED']:.2f}[s]")

    def log_last_checkpoint_location(_, latest_checkpoint):
        logger.info(f"Latest checkpoint saved as: {latest_checkpoint.last_checkpoint}")

    def log_best_checkpoint_location(_, best_checkpoint):
        logger.info(f"Best metric checkpoint saved as: {best_checkpoint.last_checkpoint}")

    trainer.add_event_handler(Events.COMPLETED, log_last_checkpoint_location, latest_checkpoint)
    trainer.add_event_handler(Events.COMPLETED, log_best_checkpoint_location, best_checkpoint)

    return trainer

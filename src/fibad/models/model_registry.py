import logging
from pathlib import Path
from typing import Any, cast

import torch.nn as nn

from fibad.plugin_utils import get_or_load_class, update_registry

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type[nn.Module]] = {}


def _torch_save(self: nn.Module, save_path: Path):
    import torch

    torch.save(self.state_dict(), save_path)


def _torch_load(self: nn.Module, load_path: Path):
    import torch

    state_dict = torch.load(load_path, weights_only=True)
    self.load_state_dict(state_dict, assign=True)


def _torch_criterion(self: nn.Module):
    """Load the criterion class using the name defined in the config and
    instantiate it with the arguments defined in the config."""

    config = cast(dict[str, Any], self.config)

    # Load the class and get any parameters from the config dictionary
    criterion_cls = get_or_load_class(config["criterion"])
    criterion_name = config["criterion"]["name"]

    arguments = {}
    if criterion_name in config:
        arguments = config[criterion_name]

    # Print some information about the criterion function and parameters used
    log_string = f"Using criterion: {criterion_name} "
    if arguments:
        log_string += f"with arguments: {arguments}."
    else:
        log_string += "with default arguments."
    logger.info(log_string)

    return criterion_cls(**arguments)


def _torch_optimizer(self: nn.Module):
    """Load the optimizer class using the name defined in the config and
    instantiate it with the arguments defined in the config."""

    config = cast(dict[str, Any], self.config)

    # Load the class and get any parameters from the config dictionary
    optimizer_cls = get_or_load_class(config["optimizer"])
    optimizer_name = config["optimizer"]["name"]

    arguments = {}
    if optimizer_name in config:
        arguments = config[optimizer_name]

    # Print some information about the optimizer function and parameters used
    log_string = f"Using optimizer: {optimizer_name} "
    if arguments:
        log_string += f"with arguments: {arguments}."
    else:
        log_string += "with default arguments."
    logger.info(log_string)

    return optimizer_cls(self.parameters(), **arguments)


def fibad_model(cls):
    """Decorator to register a model with the model registry, and to add common interface functions

    Returns
    -------
    type
        The class with additional interface functions.
    """

    if issubclass(cls, nn.Module):
        cls.save = _torch_save
        cls.load = _torch_load
        cls._criterion = _torch_criterion if not hasattr(cls, "_criterion") else cls._criterion
        cls._optimizer = _torch_optimizer if not hasattr(cls, "_optimizer") else cls._optimizer

    original_init = cls.__init__

    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.criterion = self._criterion()
        self.optimizer = self._optimizer()

    cls.__init__ = wrapped_init

    required_methods = ["train_step", "forward", "__init__"]
    for name in required_methods:
        if not hasattr(cls, name):
            logger.error(f"Fibad model {cls.__name__} missing required method {name}.")

    update_registry(MODEL_REGISTRY, cls.__name__, cls)
    return cls


def fetch_model_class(runtime_config: dict) -> type[nn.Module]:
    """Fetch the model class from the model registry.

    Parameters
    ----------
    runtime_config : dict
        The runtime configuration dictionary.

    Returns
    -------
    type
        The model class.

    Raises
    ------
    ValueError
        If a built in model was requested, but not found in the model registry.
    ValueError
        If no model was specified in the runtime configuration.
    """

    model_config = runtime_config["model"]
    model_cls = None

    try:
        model_cls = cast(type[nn.Module], get_or_load_class(model_config, MODEL_REGISTRY))
    except ValueError as exc:
        raise ValueError("Error fetching model class") from exc

    return model_cls

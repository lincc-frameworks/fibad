import importlib

import torch

from fibad.config_utils import get_runtime_config
from fibad.models import *  # noqa: F403


def run(args):
    """Placeholder for training code.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line arguments.
    """

    runtime_config = get_runtime_config(args.runtime_config)

    model_cls = _fetch_model_class(runtime_config)
    model = model_cls()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        # ~ PyTorch docs indicate that batch size should be < number of GPUs.

        # ~ PyTorch documentation recommends using torch.nn.parallel.DistributedDataParallel
        # ~ instead of torch.nn.DataParallel for multi-GPU training.
        # ~ See: https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead
        model = torch.nn.DataParallel(model)

    model.to(device)

    print("Prending to run training...")
    print(f"Runtime config: {args.runtime_config}")


def _fetch_model_class(runtime_config: dict) -> type:
    """Fetch the model class from the model registry.

    Parameters
    ----------
    runtime_config : dict
        The runtime configuration.

    Returns
    -------
    type
        The model class.
    """

    training_config = runtime_config.get("train", {})

    # if the user requests one of the built in models by name, use that
    if "model_name" in training_config:
        model_name = training_config.get("model_name", None)

        if model_name not in MODEL_REGISTRY:  # noqa: F405
            raise ValueError(f"Model not found in model registry: {model_name}")

        model_cls = MODEL_REGISTRY[model_name]  # noqa: F405

    # if the user provides a custom model, use that
    elif "model_cls" in training_config:
        model_cls = _import_module_from_string(training_config["model_cls"])

    return model_cls


def _import_module_from_string(module_path: str) -> type:
    """Dynamically import a module from a string.

    Parameters
    ----------
    module_path : str
        The import spec for the model class. Should be of the form:
        "module.submodule.class_name"

    Returns
    -------
    model_cls : type
        The model class.
    """
    module_name, class_name = module_path.rsplit(".", 1)
    model_cls = None

    # ~ Will want to do this check for each of the parent modules.
    # ~ i.e. module, module.submodule, module.submodule.subsubmodule, etc.
    if (importlib.util.find_spec(module_name)) is not None:
        module = importlib.import_module(module_name)
        if hasattr(module, class_name):
            model_cls = getattr(module, class_name)
        else:
            print(f"Model class {class_name} not found in module {module_name}")
    else:
        print(f"Module {module_name} not found")

    return model_cls

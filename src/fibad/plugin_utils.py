import importlib

from fibad.models import *  # noqa: F403
from fibad.models import MODEL_REGISTRY


def fetch_model_class(runtime_config: dict) -> type:
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

    training_config = runtime_config.get("train", {})
    model_cls = None

    # User specifies one of the built in models by name
    if "model_name" in training_config:
        model_name = training_config.get("model_name", None)

        if model_name not in MODEL_REGISTRY:  # noqa: F405
            raise ValueError(f"Model not found in model registry: {model_name}")

        model_cls = MODEL_REGISTRY[model_name]  # noqa: F405

    # User provides a custom model, attempt to import it with the module spec
    elif "model_cls" in training_config:
        model_cls = _import_module_from_string(training_config["model_cls"])

    # User failed to define a model to load
    else:
        raise ValueError("No model specified in the runtime configuration")

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

    Raises
    ------
    AttributeError
        If the model class is not found in the module that is loaded.
    ModuleNotFoundError
        If the module is not found using the provided import spec.
    """

    module_name, class_name = module_path.rsplit(".", 1)
    model_cls = None

    try:
        # Attempt to find the module spec, i.e. `module.submodule.`.
        # Will raise exception if `submodule`, 'subsubmodule', etc. is not found.
        importlib.util.find_spec(module_name)

        # `importlib.util.find_spec()` will return None if `module` is not found.
        if (importlib.util.find_spec(module_name)) is not None:
            # Load the requested module
            module = importlib.import_module(module_name)

            # Check if the requested class is in the module
            if hasattr(module, class_name):
                model_cls = getattr(module, class_name)
            else:
                raise AttributeError(f"Model class {class_name} not found in module {module_name}")

        # Raise an exception if the base module of the spec is not found
        else:
            raise ModuleNotFoundError(f"Module {module_name} not found")

    # Exception raised when a submodule of the spec is not found
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"Module {module_name} not found") from exc

    return model_cls

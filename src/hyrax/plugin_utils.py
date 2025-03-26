import importlib
from importlib import util as importlib_util
from typing import Any, Optional, TypeVar, Union

T = TypeVar("T")


def get_or_load_class(config: dict, registry: Optional[dict[str, T]] = None) -> Union[T, Any]:
    """Given a configuration dictionary and a registry dictionary, attempt to return
    the requested class either from the registry or by dynamically importing it.

    Parameters
    ----------
    config : dict
        The configuration dictionary. Must contain the key, "name".
    registry : dict
        The registry dictionary of <class name> : <class type> pairs.

    Returns
    -------
    type
        The returned class to be instantiated

    Raises
    ------
    ValueError
        User failed to specify a class to load in the runtime configuration. No
        `name` key was found in the config.
    """

    #! Once we have confidence in the config having default values, we can remove this check
    if "name" in config:
        class_name = config["name"]

        if registry and class_name in registry:
            returned_class = registry[class_name]
        else:
            returned_class = import_module_from_string(class_name)

    # User failed to define a class to load
    else:
        raise ValueError("No class requested. Specify a `name` key in the runtime config.")

    return returned_class


def import_module_from_string(module_path: str) -> Any:
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
        importlib_util.find_spec(module_name)

        # `importlib_util.find_spec()` will return None if `module` is not found.
        if (importlib_util.find_spec(module_name)) is not None:
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


def update_registry(registry: dict, name: str, class_type: type):
    """Add a class to a given registry dictionary.

    Parameters
    ----------
    registry : dict
        The registry to update.
    name : str
        The name of the class.
    class_type : type
        The class type to be instantiated.
    """

    registry.update({name: class_type})

import importlib


def get_or_load_class(config: dict, registry: dict) -> type:
    """Given a configuration dictionary and a registry dictionary, attempt to return
    the requested class either from the registry or by dynamically importing it.

    Parameters
    ----------
    config : dict
        The configuration dictionary. Should at least one of the following two
        keys: "name" or "external_cls".
    registry : dict
        The registry dictionary of <class name> : <class type> pairs.

    Returns
    -------
    type
        The returned class to be instantiated

    Raises
    ------
    ValueError
        User specified a `name` key in the config that doesn't match any keys in the registry.
    ValueError
        User failed to specify a class to load in the runtime configuration. Neither
        a `name` nor `external_cls` key was found in the config.
    """

    # User specifies one of the built in classes by name
    if "name" in config:
        class_name = config.get("name")

        if class_name not in registry:
            raise ValueError(f"Could not find {class_name} in registry: {registry.keys()}")

        returned_class = registry[class_name]

    # User provides an external class, attempt to import it with the module spec
    elif "external_cls" in config:
        returned_class = import_module_from_string(config["external_cls"])

    # User failed to define a class to load
    else:
        raise ValueError("No class requested. Specify a `name` or `external_cls` key in the runtime config.")

    return returned_class


def import_module_from_string(module_path: str) -> type:
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

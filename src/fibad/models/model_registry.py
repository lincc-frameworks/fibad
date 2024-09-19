from fibad.plugin_utils import get_or_load_class, update_registry

MODEL_REGISTRY = {}


def fibad_model(cls):
    """Decorator to register a model with the model registry.

    Returns
    -------
    type
        The original, unmodified class.
    """
    update_registry(MODEL_REGISTRY, cls.__name__, cls)
    return cls


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

    model_config = runtime_config["model"]
    model_cls = None

    try:
        model_cls = get_or_load_class(model_config, MODEL_REGISTRY)
    except ValueError as exc:
        raise ValueError("Error fetching model class") from exc

    return model_cls

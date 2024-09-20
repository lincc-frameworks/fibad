from fibad.plugin_utils import get_or_load_class, update_registry

DATA_LOADER_REGISTRY = {}


def fibad_data_loader(cls):
    """Decorator to register a data loader with the registry.

    Returns
    -------
    type
        The original, unmodified class.
    """
    update_registry(DATA_LOADER_REGISTRY, cls.__name__, cls)
    return cls


def fetch_data_loader_class(runtime_config: dict) -> type:
    """Fetch the data loader class from the registry.

    Parameters
    ----------
    runtime_config : dict
        The runtime configuration dictionary.

    Returns
    -------
    type
        The data loader class.

    Raises
    ------
    ValueError
        If a built in data loader was requested, but not found in the registry.
    ValueError
        If no data loader was specified in the runtime configuration.
    """

    data_loader_config = runtime_config["data_loader"]
    data_loader_cls = None

    try:
        data_loader_cls = get_or_load_class(data_loader_config, DATA_LOADER_REGISTRY)
    except ValueError as exc:
        raise ValueError("Error fetching data loader class") from exc

    return data_loader_cls

import logging

from torch.utils.data import Dataset

from hyrax.plugin_utils import get_or_load_class, update_registry

logger = logging.getLogger(__name__)

DATA_SET_REGISTRY: dict[str, type[Dataset]] = {}


def hyrax_data_set(cls):
    """Decorator to register a data loader with the registry.

    Returns
    -------
    type
        The original, unmodified class.
    """
    required_methods = ["shape", "__getitem__", "__len__", "ids"]
    for name in required_methods:
        if not hasattr(cls, name):
            logger.error(f"Hyrax data set {cls.__name__} missing required method {name}.")

    update_registry(DATA_SET_REGISTRY, cls.__name__, cls)
    return cls


def fetch_data_set_class(runtime_config: dict) -> type[Dataset]:
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

    data_set_config = runtime_config["data_set"]
    data_set_cls = None

    try:
        data_set_cls = get_or_load_class(data_set_config, DATA_SET_REGISTRY)
    except ValueError as exc:
        raise ValueError("Error fetching data set class") from exc

    return data_set_cls

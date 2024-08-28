import logging

import ignite.distributed as idist
from ignite.engine import Engine

from fibad.data_loaders.data_loader_registry import fetch_data_loader_class
from fibad.models.model_registry import fetch_model_class

logger = logging.getLogger(__name__)


def run(config):
    """Placeholder for training code.

    Parameters
    ----------
    config : dict
        The parsed config file as a nested
        dict
    """

    data_loader_cls = fetch_data_loader_class(config)
    fibad_data_loader = data_loader_cls(config.get("data_loader", {}))
    data_set = fibad_data_loader.data_set()
    data_loader = _train_data_loader(data_set, config.get("data_loader", {}))

    model_cls = fetch_model_class(config)
    model = model_cls(model_config=config.get("model", {}), shape=fibad_data_loader.shape())

    model = idist.auto_model(model)

    trainer = Engine(model.train_step)
    trainer.run(data_loader, max_epochs=config.get("model", {}).get("epochs", 2))

    logger.info("Finished Training")


def _train_data_loader(data_set, config):
    # ~ idist.auto_dataloader will accept a **kwargs parameter, and pass values
    # ~ through to the underlying pytorch DataLoader.
    data_loader = idist.auto_dataloader(
        data_set,
        batch_size=config.get("batch_size", 4),
        shuffle=config.get("shuffle", True),
        drop_last=config.get("drop_last", False),
    )

    return data_loader

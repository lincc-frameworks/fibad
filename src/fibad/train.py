import logging

import ignite.distributed as idist
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine

from fibad.data_loaders.data_loader_registry import fetch_data_loader_class
from fibad.models.model_registry import fetch_model_class

logger = logging.getLogger(__name__)


def run(config):
    """Run the training process for a given model and data loader.

    Parameters
    ----------
    config : dict
        The parsed config file as a nested
        dict
    """

    # Fetch data loader class specified in config and create an instance of it
    data_loader_cls = fetch_data_loader_class(config)
    data_loader = data_loader_cls(config.get("data_loader", {}))

    # Get the pytorch.dataset from dataloader, and use it to create a distributed dataloader
    data_set = data_loader.data_set()
    dist_data_loader = _train_data_loader(data_set, config.get("data_loader", {}))

    # Fetch model class specified in config and create an instance of it
    model_cls = fetch_model_class(config)
    model = model_cls(model_config=config.get("model", {}), shape=data_loader.shape())

    # Create trainer, a pytorch-ignite `Engine` object
    trainer = _create_trainer(model)

    # Run the training process
    trainer.run(dist_data_loader, max_epochs=config.get("model", {}).get("epochs", 2))

    # Save the trained model
    model.save()
    logger.info("Finished Training")


def _train_data_loader(data_set, config):
    # ~ idist.auto_dataloader will accept a **kwargs parameter, and pass values
    # ~ through to the underlying pytorch DataLoader.
    # ~ Currently, our config includes unexpected keys like `name`, that cause
    # ~ an exception. It would be nice to reduce this to:
    # ~ `data_loader = idist.auto_dataloader(data_set, **config)`
    data_loader = idist.auto_dataloader(
        data_set,
        batch_size=config.get("batch_size", 4),
        shuffle=config.get("shuffle", True),
        num_workers=config.get("num_workers", 2),
    )

    return data_loader


def _create_trainer(model):
    device = idist.device()
    print(f"Working with device: {device}")
    model = idist.auto_model(model)

    def train_step(engine, batch):
        batch = tuple(i.to(device) for i in batch)
        model.train_step(engine, batch)

    trainer = Engine(train_step)
    ProgressBar().attach(trainer)

    return trainer

import logging

import torch

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
    data_loader = fibad_data_loader.get_data_loader()

    model_cls = fetch_model_class(config)
    model = model_cls(model_config=config.get("model", {}), shape=fibad_data_loader.shape())

    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()

    # We don't expect mps (Apple's Metal backend) and cuda (Nvidia's backend) to ever be
    # both available on the same system.
    device_str = "cuda:0" if cuda_available else "cpu"
    device_str = "mps" if mps_available else "cpu"

    logger.info(f"Initializing torch with device string {device_str}")

    device = torch.device(device_str)
    if torch.cuda.device_count() > 1:
        # ~ PyTorch docs indicate that batch size should be < number of GPUs.

        # ~ PyTorch documentation recommends using torch.nn.parallel.DistributedDataParallel
        # ~ instead of torch.nn.DataParallel for multi-GPU training.
        # ~ See: https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead
        model = torch.nn.DataParallel(model)

    model.to(device)

    model.train(data_loader, device=device)

    model.save()
    logger.info("Finished Training")

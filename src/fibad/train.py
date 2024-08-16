import torch

from fibad.data_loaders.data_loader_registry import fetch_data_loader_class
from fibad.models.model_registry import fetch_model_class


def run(config):
    """Placeholder for training code.

    Parameters
    ----------
    config : dict
        The parsed config file as a nested
        dict
    """

    model_cls = fetch_model_class(config)
    model = model_cls(config.get("model", {}))

    data_loader_cls = fetch_data_loader_class(config)
    data_loader = data_loader_cls(config.get("data_loader", {})).get_data_loader()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        # ~ PyTorch docs indicate that batch size should be < number of GPUs.

        # ~ PyTorch documentation recommends using torch.nn.parallel.DistributedDataParallel
        # ~ instead of torch.nn.DataParallel for multi-GPU training.
        # ~ See: https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead
        model = torch.nn.DataParallel(model)

    model.to(device)

    model.train(data_loader, device=device)

    model.save()
    print("Finished Training")

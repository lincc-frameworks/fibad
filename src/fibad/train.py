import torch

from fibad.config_utils import get_runtime_config
from fibad.data_loaders.data_loader_registry import fetch_data_loader_class
from fibad.models.model_registry import fetch_model_class


def run(args, config):
    """Placeholder for training code.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line arguments.

    config : dict
        The parsed config file as a nested
        dict
    """

    runtime_config = get_runtime_config(args.runtime_config)

    model_cls = fetch_model_class(runtime_config)
    model = model_cls(runtime_config.get("model", {}))

    data_loader_cls = fetch_data_loader_class(runtime_config)
    data_loader = data_loader_cls(runtime_config.get("data_loader", {})).get_data_loader()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        # ~ PyTorch docs indicate that batch size should be < number of GPUs.

        # ~ PyTorch documentation recommends using torch.nn.parallel.DistributedDataParallel
        # ~ instead of torch.nn.DataParallel for multi-GPU training.
        # ~ See: https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead
        model = torch.nn.DataParallel(model)

    model.to(device)

    model.train(data_loader)

    model.save()
    print("Finished Training")

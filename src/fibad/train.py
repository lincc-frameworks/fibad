import torch

from fibad.config_utils import get_runtime_config
from fibad.plugin_utils import fetch_model_class


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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        # ~ PyTorch docs indicate that batch size should be < number of GPUs.

        # ~ PyTorch documentation recommends using torch.nn.parallel.DistributedDataParallel
        # ~ instead of torch.nn.DataParallel for multi-GPU training.
        # ~ See: https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead
        model = torch.nn.DataParallel(model)

    model.to(device)

    data_set = model.data_set()
    data_loader = model.data_loader(data_set)

    model.train(data_loader)

    model.save()
    print("Finished Training")

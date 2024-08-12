import torch

from fibad.config_utils import get_runtime_config
from fibad.plugin_utils import fetch_model_class


def run(args):
    """Placeholder for training code.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line arguments.
    """

    runtime_config = get_runtime_config(args.runtime_config)

    model_cls = fetch_model_class(runtime_config)
    model = model_cls()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        # ~ PyTorch docs indicate that batch size should be < number of GPUs.

        # ~ PyTorch documentation recommends using torch.nn.parallel.DistributedDataParallel
        # ~ instead of torch.nn.DataParallel for multi-GPU training.
        # ~ See: https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead
        model = torch.nn.DataParallel(model)

    model.to(device)

    training_config = runtime_config.get("train", {})

    model.save(training_config.get("model_weights_filepath"))
    print("Finished Training")

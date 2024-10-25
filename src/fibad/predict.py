import logging
from pathlib import Path

import numpy as np
from torch import Tensor

from fibad.config_utils import ConfigDict, create_results_dir, log_runtime_config
from fibad.pytorch_ignite import create_evaluator, dist_data_loader, setup_model_and_dataset

logger = logging.getLogger(__name__)


def run(config: ConfigDict):
    """Run Prediction

    Parameters
    ----------
    config : ConfigDict
        The parsed config file as a nested dict
    """

    model, data_set = setup_model_and_dataset(config, split=config["predict"]["split"])
    logger.info(f"data set has length {len(data_set)}")
    data_loader = dist_data_loader(data_set, config)

    # Create a results directory and dump our config there
    results_dir = create_results_dir(config, "predict")
    log_runtime_config(config, results_dir)
    load_model_weights(config, model)

    write_index = 0
    object_ids = list(data_set.ids() if hasattr(data_set, "ids") else range(len(data_set)))

    def _save_batch(batch_results: Tensor):
        """Receive and write results tensors to results_dir immediately
        This function writes a single numpy binary file for each object.
        """
        nonlocal write_index
        batch_results = batch_results.detach().to("cpu")
        for tensor in batch_results:
            object_id = object_ids[write_index]
            filename = f"{object_id}.npy"
            savepath = results_dir / filename
            if savepath.exists():
                RuntimeError("The path to save results for object {object_id} already exists.")
            np.save(savepath, tensor.numpy(), allow_pickle=False)
            write_index += 1

    evaluator = create_evaluator(model, _save_batch)
    evaluator.run(data_loader)
    logger.info("finished evaluating...")


def load_model_weights(config: ConfigDict, model):
    """Loads the model weights from a file. Raises RuntimeError if this is not possible due to
    config, missing or malformed file

    Parameters
    ----------
    config : ConfigDict
        Full runtime configuration
    model : _type_
        The model class to load weights into

    """
    weights_file = config["predict"]["model_weights_file"]

    if not weights_file:
        # TODO: Look at the last predict run from the rundir
        # use config["model"]["weights_filename"] to find the weights
        # Proceed with those weights
        raise RuntimeError("Must define pretrained_model in the predict section of fibad config.")

    weights_file = Path(weights_file)

    if not weights_file.exists():
        raise RuntimeError(f"Model Weights file {weights_file} does not exist")

    try:
        model.load(weights_file)
    except Exception as err:
        msg = f"Model weights file {weights_file} did not load properly. Are you sure you are predicting "
        msg += "using the correct model"
        raise RuntimeError(msg) from err

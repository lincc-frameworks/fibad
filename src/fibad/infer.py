import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from tensorboardX import SummaryWriter
from torch import Tensor

from fibad.config_utils import (
    ConfigDict,
    create_results_dir,
    find_most_recent_results_dir,
    log_runtime_config,
)
from fibad.data_sets.inference_dataset import InferenceDataSetWriter
from fibad.pytorch_ignite import (
    create_evaluator,
    dist_data_loader,
    setup_dataset,
    setup_model,
)
from fibad.vector_dbs.vector_db_factory import vector_db_factory

logger = logging.getLogger(__name__)


def run(config: ConfigDict):
    """Run inference on a model using a dataset

    Parameters
    ----------
    config : ConfigDict
        The parsed config file as a nested dict
    """
    context = {}

    # Create a results directory and dump our config there
    results_dir = create_results_dir(config, "infer")

    # Create a tensorboardX logger
    tensorboardx_logger = SummaryWriter(log_dir=results_dir)

    data_set = setup_dataset(config, tensorboardx_logger)

    model = setup_model(config, data_set)
    logger.info(f"data set has length {len(data_set)}")  # type: ignore[arg-type]
    data_loader = dist_data_loader(data_set, config, split=config["infer"]["split"])

    log_runtime_config(config, results_dir)
    load_model_weights(config, model)
    context["results_dir"] = results_dir

    vector_db = vector_db_factory(config, context)
    if vector_db:
        vector_db.create()

    data_writer = InferenceDataSetWriter(data_set, results_dir)

    # These are values the _save_batch callback needs to run
    write_index = 0
    batch_index = 0
    object_ids = list(int(id) for id in data_set.ids())  # type: ignore[attr-defined]

    def _save_batch(batch_results: Tensor):
        """Receive and write results tensors to results_dir immediately
        This function writes a single numpy binary file for each object.
        """
        nonlocal write_index
        nonlocal batch_index
        nonlocal object_ids
        nonlocal data_writer

        batch_len = len(batch_results)
        batch_results = batch_results.detach().to("cpu")
        batch_object_ids = [object_ids[id] for id in range(write_index, write_index + len(batch_results))]

        # Save results to vector database
        nonlocal vector_db
        if vector_db:
            ids: list[str | int] = [str(id) for id in batch_object_ids]
            vectors: list[np.ndarray] = [t.flatten().numpy() for t in batch_results]
            vector_db.insert(ids=ids, vectors=vectors)

        # Save results from this batch in a numpy file as a structured array
        data_writer.write_batch(np.array(batch_object_ids), [t.numpy() for t in batch_results])
        write_index += batch_len

    # Run inference
    evaluator = create_evaluator(model, _save_batch)
    evaluator.run(data_loader)

    # Write out a dictionary to map IDs->Batch
    data_writer.write_index()

    # Write out our tensorboard stuff
    tensorboardx_logger.close()

    # Log completion
    logger.info(f"Inference results saved in: {results_dir}")


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
    weights_file: Optional[Union[str, Path]] = (
        config["infer"]["model_weights_file"] if config["infer"]["model_weights_file"] else None
    )

    if weights_file is None:
        recent_results_path = find_most_recent_results_dir(config, "train")
        if recent_results_path is None:
            raise RuntimeError("Must define model_weights_file in the [infer] section of fibad config.")

        weights_file = recent_results_path / config["train"]["weights_filepath"]

    # Ensure weights file is a path object.
    weights_file_path = Path(weights_file)

    if not weights_file_path.exists():
        raise RuntimeError(f"Model Weights file {weights_file_path} does not exist")

    try:
        model.load(weights_file_path)
    except Exception as err:
        msg = (
            f"Model weights file {weights_file_path} did not load properly. Are you sure you are predicting "
        )
        msg += "using the correct model"
        raise RuntimeError(msg) from err

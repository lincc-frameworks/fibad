import logging
from pathlib import Path
from typing import Optional, Union

import chromadb
import numpy as np
from torch import Tensor

from fibad.config_utils import (
    ConfigDict,
    create_results_dir,
    find_most_recent_results_dir,
    log_runtime_config,
)
from fibad.pytorch_ignite import (
    create_evaluator,
    dist_data_loader,
    setup_dataset,
    setup_model,
)

logger = logging.getLogger(__name__)


def run(config: ConfigDict):
    """Run inference on a model using a dataset

    Parameters
    ----------
    config : ConfigDict
        The parsed config file as a nested dict
    """

    data_set = setup_dataset(config, split=config["infer"]["split"])
    model = setup_model(config, data_set)
    logger.info(f"data set has length {len(data_set)}")  # type: ignore[arg-type]
    data_loader = dist_data_loader(data_set, config, split=config["infer"]["split"])

    # Create a results directory and dump our config there
    results_dir = create_results_dir(config, "infer")
    log_runtime_config(config, results_dir)
    load_model_weights(config, model)

    # Create a chromadb in results dir
    if config["infer"]["chromadb"]:
        chromadb_client = chromadb.PersistentClient(path=str(results_dir))
        collection = chromadb_client.create_collection(
            name="fibad",
            metadata={
                # These are all chromdb defaults. We may want to configure them
                "hsnw:space": "l2",
                "hsnw:construction_ef": 100,
                "hsnw:search_ef": 100,
            },
        )

    write_index = 0
    batch_index = 0
    object_ids: list[int] = list(int(data_set.ids()) if hasattr(data_set, "ids") else range(len(data_set)))  # type: ignore[arg-type]

    def _save_batch(batch_results: Tensor):
        """Receive and write results tensors to results_dir immediately
        This function writes a single numpy binary file for each object.
        """
        nonlocal write_index
        nonlocal batch_index
        nonlocal object_ids

        batch_len = len(batch_results)
        batch_results = batch_results.detach().to("cpu")
        batch_object_ids = [object_ids[id] for id in range(write_index, write_index + len(batch_results))]

        # Save results to ChromaDB vector database
        if config["infer"]["chromadb"]:
            nonlocal collection
            chroma_ids: list[str] = [str(id) for id in batch_object_ids]
            embeddings: list[np.ndarray] = [t.flatten().numpy() for t in batch_results]
            collection.add(embeddings=embeddings, ids=chroma_ids)

        # Save results from this batch in a numpy file as a structured array
        first_tensor = batch_results[0].numpy()
        structured_batch_type = np.dtype(
            [("id", np.int64), ("tensor", first_tensor.dtype, first_tensor.shape)]
        )
        structured_batch = np.zeros(batch_len, structured_batch_type)
        structured_batch["id"] = batch_object_ids
        structured_batch["tensor"] = [t.numpy() for t in batch_results]

        filename = f"batch_{batch_index}.npy"
        savepath = results_dir / filename
        if savepath.exists():
            RuntimeError("The path to save results for object {object_id} already exists.")

        np.save(savepath, structured_batch, allow_pickle=False)

        batch_index += 1
        write_index += batch_len

    evaluator = create_evaluator(model, _save_batch)
    evaluator.run(data_loader)

    logger.info(f"Results saved in {results_dir}")
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

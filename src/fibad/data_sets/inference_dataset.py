import numpy as np
from typing import Optional, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .data_set_registry import fibad_data_set

@fibad_data_set
class InferenceDataSet(Dataset):
    """This is a dataset class to represent the situations where we wish to treat the output of inference 
        as a dataset. e.g. when performing umap/visualization operations"""

    def __init__(self, config, split: Union[str, bool], results_dir = Optional[Union[Path, str]] = None ):
        """ Initialize an inference results directory as a data source. Accepts an override of what 
        directory to use"""
        if results_dir is None:
            if self.config["results"]["inference_dir"]:
                results_dir = self.config["results"]["inference_dir"]
            else:
                results_dir = find_most_recent_results_dir(self.config, verb="infer")
                msg = f"Using most recent results dir {results_dir} for lookup."
                msg += "Use the [results] inference_dir config to set a directory or pass it to this verb."
                logger.info(msg)

        if results_dir is None:
            msg = "Could not find a results directory. Run infer or use "
            msg += "[results] inference_dir config to specify a directory"
            logger.error(msg)
            return None

        if isinstance(results_dir, str):
            results_dir = Path(results_dir)

        # Open the batch index numpy file.
        # Loop over files and create if it does not exist
        batch_index_path = results_dir / "batch_index.npy"
        if not batch_index_path.exists():
            self._create_index(results_dir)

        batch_index = np.load(results_dir / "batch_index.npy")

        

    def shape(self):
        """The shape of the dataset (discoverable from files)"""
        pass

    def ids(self):
        pass

    def __getitem__(self, idx:int) -> torch.Tensor:
        pass

    def __len__(self) -> int:
        pass

    def _create_index(self, results_dir: Path):
        """Recreate the index into the batch numpy files

        Parameters
        ----------
        results_dir : Path
            Path to the batch numpy files
        """
        ids = []
        batch_nums = []
        # Use the batched numpy files to assemble an index.
        logger.info("Recreating index...")
        for file in results_dir.glob("batch_*.npy"):
            print(".", end="", flush=True)
            m = re.match(r"batch_([0-9]+).npy", file.name)
            if m is None:
                logger.warn(f"Could not find batch number for {file}")
                continue
            batch_num = int(m[1])
            recarray = np.load(file)
            ids += list(recarray["id"])
            batch_nums += [batch_num] * len(recarray["id"])

        save_batch_index(results_dir, np.array(ids), np.array(batch_nums))
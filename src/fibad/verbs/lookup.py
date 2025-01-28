import logging
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Union

import numpy as np

from fibad.config_utils import find_most_recent_results_dir
from fibad.infer import save_batch_index

from .verb_registry import Verb, fibad_verb

logger = logging.getLogger(__name__)


@fibad_verb
class Lookup(Verb):
    """Look up an inference result using the ID of a data member"""

    cli_name = "lookup"
    add_parser_kwargs = {}

    @staticmethod
    def setup_parser(parser: ArgumentParser):
        """Set up our arguments by configuring a subparser

        Parameters
        ----------
        parser : ArgumentParser
            The sub-parser to configure
        """
        parser.add_argument("-i", "--id", type=str, required=True, help="ID of image")
        parser.add_argument(
            "-r", "--results-dir", type=str, required=False, help="Directory containing inference results."
        )

    def run_cli(self, args: Optional[Namespace] = None):
        """Entrypoint to Lookup from the CLI.

        Parameters
        ----------
        args : Optional[Namespace], optional
            The parsed command line arguments

        """
        logger.info("Lookup run from cli")
        if args is None:
            raise RuntimeError("Run CLI called with no arguments.")
        # This is where we map from CLI parsed args to a
        # self.run (args) call.
        vector = self.run(id=args.id, results_dir=args.results_dir)
        if vector is None:
            logger.info("No inference result found")
        else:
            logger.info("Inference result found")
            print(vector)

    def run(self, id: str, results_dir: Optional[Union[Path, str]] = None) -> Optional[np.ndarray]:
        """Lookup the latent-space representation of a particular ID

        Requires the relevant dataset to be configured, and for inference to have been run.

        Parameters
        ----------
        id : str
            The ID of the input data to look up the inference result

        results_dir : str, Optional
            The directory containing the inference results.

        Returns
        -------
        Optional[np.ndarray]
            The output tensor of the model for the given input.
        """
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
            self.create_index(results_dir)

        batch_index = np.load(results_dir / "batch_index.npy")
        batch_num = batch_index[batch_index["id"] == int(id)]["batch_num"]
        if len(batch_num) == 0:
            return None
        batch_num = batch_num[0]

        recarray = np.load(results_dir / f"batch_{batch_num}.npy")
        tensor = recarray[recarray["id"] == int(id)]["tensor"]
        if len(tensor) == 0:
            return None

        return np.array(tensor[0])

    def create_index(self, results_dir: Path):
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

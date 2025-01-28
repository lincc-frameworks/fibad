import logging
from argparse import ArgumentParser, Namespace
from typing import Optional, Union
from pathlib import Path
import umap

from .verb_registry import Verb, fibad_verb
from fibad.config_utils import find_most_recent_results_dir

logger = logging.getLogger(__name__)

@fibad_verb
class Umap(Verb):
    """Stub of visualization verb"""

    cli_name = "umap"
    add_parser_kwargs = {}

    @staticmethod
    def setup_parser(parser: ArgumentParser):
        """Stub of parser setup"""
        parser.add_argument(
            "-r", "--results-dir", type=str, required=False, help="Directory containing inference results."
        )

    # Should there be a version of this on the base class which uses a dict on the Verb
    # superclass to build the call to run based on what the subclass verb defined in setup_parser
    def run_cli(self, args: Optional[Namespace] = None):
        """Stub CLI implementation"""
        logger.info("Search run from cli")
        if args is None:
            raise RuntimeError("Run CLI called with no arguments.")
        # This is where we map from CLI parsed args to a
        # self.run (args) call.
        return self.run(results_dir=args.results_dir)

    def run(self, results_dir: Optional[Union[Path, str]], **kwargs):
        """Create visualization"""
        # Lookup latest infer directory, or use the provided one.
        # TODO: duplicated from lookup
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

        # TODO pass in kwargs so people can control umap?
        #      Should this be config or args?
        reducer = umap.UMAP()

        # Load all the latent space data.
        # TODO sample this based on a config

        # Fit a single reducer on the sampled data
        reducer.fit()

        # Save the reducer to our results dir

        # Run all data through the reducer in batches
        # writing out as we go



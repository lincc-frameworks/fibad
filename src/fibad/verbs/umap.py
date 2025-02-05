import logging
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Union

import numpy as np
import umap

from fibad.config_utils import create_results_dir
from fibad.data_sets.inference_dataset import InferenceDataSet, InferenceDataSetWriter

from .verb_registry import Verb, fibad_verb

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
            "-i",
            "--input-dir",
            type=str,
            required=False,
            help="Directory containing inference results to umap.",
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
        return self.run(input_dir=args.input_dir)

    def run(self, input_dir: Optional[Union[Path, str]] = None, **kwargs):
        """Create a umap of a particular inference run"""

        # TODO pass in kwargs so people can control umap?
        #      Should this be config or args?
        reducer = umap.UMAP(**kwargs)

        # Set up the results directory where we will store our umapped output
        results_dir = create_results_dir(self.config, "umap")
        umap_results = InferenceDataSetWriter(results_dir)

        # Load all the latent space data.
        inference_results = InferenceDataSet(self.config, split=False, results_dir=input_dir)
        total_length = len(inference_results)

        # Sample the data to fit
        config_sample_size = self.config["umap"]["fit_sample_size"]
        sample_size = np.min([config_sample_size if config_sample_size else np.inf, total_length])
        rng = np.random.default_rng()
        index_choices = rng.choice(np.arange(total_length), size=sample_size, replace=False)
        data_sample = inference_results[index_choices].numpy()

        # Fit a single reducer on the sampled data
        reducer.fit(data_sample)

        # Save the reducer to our results directory
        with open(results_dir / "umap.pickle", "wb") as f:
            pickle.dump(reducer, f)

        # Run all data through the reducer in batches, writing it out as we go.
        batch_size = self.config["data_loader"]["batch_size"]
        num_batches = int(np.ceil(total_length / batch_size))

        all_indexes = np.arange(0, total_length)
        all_ids = np.array([int(i) for i in inference_results.ids()])
        for batch_indexes in np.array_split(all_indexes, num_batches):
            batch = inference_results[batch_indexes]
            batch_ids = all_ids[batch_indexes]
            transformed_batch = reducer.transform(batch)
            umap_results.write_batch(batch_ids, transformed_batch)

        umap_results.write_index()

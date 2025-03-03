import logging
import pickle
import warnings
from argparse import ArgumentParser, Namespace
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional, Union

import numpy as np

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
        logger.info("umap run from cli")
        if args is None:
            raise RuntimeError("Run CLI called with no arguments.")

        # This is where we map from CLI parsed args to a
        # self.run (args) call.
        return self.run(input_dir=args.input_dir)

    def run(self, input_dir: Optional[Union[Path, str]] = None):
        """
        Create a umap of a particular inference run

        This method loads the latent space representations from an inference run,
        samples a subset of data points, flattens them if necessary, and then fits
        a UMAP model. The fitted reducer is then used to transform the entire dataset
        into a lower-dimensional space.

        Parameters
        ----------
        input_dir : str or Path, Optional
            The directory containing the inference results.

        Returns
        -------
        None
            The method does not return anything but saves the UMAP representations to disk.
        """
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            return self._run(input_dir)

    def _run(self, input_dir: Optional[Union[Path, str]] = None):
        """See run()"""
        from multiprocessing import Pool

        import umap
        from tqdm.auto import tqdm

        from fibad.config_utils import create_results_dir
        from fibad.data_sets.inference_dataset import InferenceDataSet, InferenceDataSetWriter

        self.reducer = umap.UMAP(**self.config["umap.UMAP"])

        # Set up the results directory where we will store our umapped output
        results_dir = create_results_dir(self.config, "umap")
        logger.info(f"Saving UMAP results to {results_dir}")
        umap_results = InferenceDataSetWriter(results_dir)

        # Load all the latent space data.
        inference_results = InferenceDataSet(self.config, results_dir=input_dir)
        total_length = len(inference_results)

        # Sample the data to fit
        config_sample_size = self.config["umap"]["fit_sample_size"]
        sample_size = np.min([config_sample_size if config_sample_size else np.inf, total_length])
        rng = np.random.default_rng()
        index_choices = rng.choice(np.arange(total_length), size=sample_size, replace=False)

        # If the input to umap is not of the shape [samples,input_dims] we reshape the input accordingly
        data_sample = inference_results[index_choices].numpy().reshape((sample_size, -1))

        # Fit a single reducer on the sampled data
        self.reducer.fit(data_sample)

        # Save the reducer to our results directory
        with open(results_dir / "umap.pickle", "wb") as f:
            pickle.dump(self.reducer, f)

        # Run all data through the reducer in batches, writing it out as we go.
        batch_size = self.config["data_loader"]["batch_size"]
        num_batches = int(np.ceil(total_length / batch_size))

        all_indexes = np.arange(0, total_length)
        all_ids = np.array([int(i) for i in inference_results.ids()])

        # Process pool to do all the transforms
        with Pool(processes=cpu_count()) as pool:
            # Generator expression that gives a batch tuple composed of:
            # batch ids, inference results
            args = (
                (
                    all_ids[batch_indexes],
                    # We flatten all dimensions of the input array except the dimension
                    # corresponding to batch elements. This ensures that all inputs to
                    # the UMAP algorithm are flattend per input item in the batch
                    inference_results[batch_indexes].reshape(len(batch_indexes), -1),
                )
                for batch_indexes in np.array_split(all_indexes, num_batches)
            )

            # iterate over the mapped results to write out the umapped points
            # imap returns results as they complete so writing should complete in parallel for large datasets
            for batch_ids, transformed_batch in tqdm(
                pool.imap(self._transform_batch, args),
                desc="Creating LowerDimensional Representation using UMAP:",
                total=num_batches,
            ):
                logger.debug("Writing a batch out async...")
                umap_results.write_batch(batch_ids, transformed_batch)

        umap_results.write_index()

    def _transform_batch(self, batch_tuple: tuple):
        """Private helper to transform a single batch

        Parameters
        ----------
        batch_tuple : tuple()
            first element is the IDs of the batch as a numpy array
            second element is the inference results to transform as a numpy array with shape (batch_len, N)
            where N is the total number of dimensions in the inference result. Caller flattens all inference
            result axes for us.

        Returns
        -------
        tuple
            first element is the ids of the batch as a numpy array
            second element is the results of running the umap transform on the input as a numpy array.
        """
        batch_ids, batch = batch_tuple
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            logger.debug("Transforming a batch ...")
            return (batch_ids, self.reducer.transform(batch))

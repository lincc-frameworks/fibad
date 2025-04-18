import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Union

import numpy as np

from hyrax.config_utils import (
    create_results_dir,
    find_most_recent_results_dir,
)
from hyrax.data_sets.inference_dataset import InferenceDataSet
from hyrax.vector_dbs.vector_db_factory import vector_db_factory

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class DatabaseInsert(Verb):
    """Stub of similarity search"""

    cli_name = "search"
    add_parser_kwargs = {}

    @staticmethod
    def setup_parser(parser: ArgumentParser):
        """No need for parser right now, may change later."""
        pass

    # If both of these move to the verb superclass then a new verb is basically
    #
    # If you want no args, just make the class, define run(self)
    # If you want args
    #     1) write setup_parser (which sets up for ArgumentParser and name/type info for cli run)
    #     2) write run(self, <your args>) to do what you want
    #

    # Should there be a version of this on the base class which uses a dict on the Verb
    # superclass to build the call to run based on what the subclass verb defined in setup_parser
    def run_cli(self, args: Optional[Namespace] = None):
        """Stub CLI implementation"""
        logger.info("Insert inference results into vector database")
        if args is None:
            raise RuntimeError("Run CLI called with no arguments.")
        # This is where we map from CLI parsed args to a
        # self.run (args) call.
        return self.run()

    def run(self):
        """Insert inference results into vector database"""
        config = self.config

        infer_results_dir: Optional[Union[str, Path]] = (
            config["vector_db"]["infer_results_dir"] if config["vector_db"]["infer_results_dir"] else None
        )

        if infer_results_dir is None:
            infer_results_dir = find_most_recent_results_dir(config, "infer")
            if infer_results_dir is None:
                raise RuntimeError(
                    "Must define infer_results_dir in the [vector_db] section of hyrax config."
                )

        inference_results_path = Path(infer_results_dir)

        vector_db_dir = None
        if config["vector_db"]["vector_db_dir"]:
            vector_db_dir = config["vector_db"]["vector_db_dir"]
        else:
            vector_db_dir = create_results_dir(config, "vector_db")
        context = {"results_dir": vector_db_dir}

        # Create an instance of the InferenceDataSet
        inference_data_set = InferenceDataSet(config, inference_results_path)

        # Create an instance of the vector database to insert into
        vector_db = vector_db_factory(config, context)
        if vector_db:
            vector_db.create()

        inference_result_ids = list(inference_data_set.ids())

        # TODO: Need to modify this to insert chunks
        for idx, vec in enumerate(inference_data_set):
            logger.debug(f"Writing Vector DB for index {idx}")
            ids: list[str | int] = str(inference_result_ids[idx])
            vectors: list[np.ndarray] = vec.flatten().numpy()
            logger.debug("Inserting vectors into database")
            vector_db.insert(ids=ids, vectors=vectors)

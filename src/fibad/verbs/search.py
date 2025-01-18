import logging
from argparse import ArgumentParser, Namespace
from typing import Optional

from fibad.config_utils import ConfigDict

from .verb_registry import Verb, fibad_verb

logger = logging.getLogger(__name__)


@fibad_verb
class Search(Verb):
    cli_name = "search"
    add_parser_kwargs = {}

    @staticmethod
    def setup_parser(parser: ArgumentParser):
        parser.add_argument("-i", "--image-file", type=str, help="Path to image file", required=True)

    # If both of these move to the verb superclass then a new verb is basically
    #
    # If you want no args, just make the class, define run(self)
    # If you want args
    #     1) write setup_parser (which sets up for ArgumentParser and name/type info for cli run)
    #     2) write run(self, <your args>) to do what you want
    #
    # Should this class be a callable class to simlify things further?

    # Should this be on the base class so we don't have to repeat it on every verb?
    def __init__(self, config: Optional[ConfigDict] = None):
        self.config = config

    # Should there be a version of this on the base class which uses a dict on the Verb
    # superclass to build the call to run based on what the subclass verb defined in setup_parser
    def run_cli(self, args: Optional[Namespace] = None):
        logger.info("Search run from cli")
        # This is where we map from CLI parsed args to a
        # self.run (args) call.
        return self.run(image_file=args.image_file)

    def run(self, image_file: str):
        """Search for... <TODO xcxc>

        Parameters
        ----------
        image_file : str
            _description_
        """
        logger.info(f"Got Image {image_file}")

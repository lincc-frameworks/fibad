import logging
import sys
from pathlib import Path
from typing import Union

from .config_utils import get_runtime_config


class Fibad:
    """
    Overall class that represents an interface into fibad. Currently this encapsulates a configuration and is
    the external interface to all verbs in a programmatic context.

    CLI functions in fibad_cli are implemented by calling this class
    """

    verbs = ["train", "predict", "download"]

    def __init__(self, *, config_file: Union[Path, str] = None):
        """Initialize fibad. Always applies the default config, and merges it with any provided config file.

        Parameters
        ----------
        config_file : Union[Path, str], optional
            filename or pathlib.Path to a config file, by default None
        """
        self.config = get_runtime_config(runtime_config_filepath=config_file)
        general_config = self.config.get("general", {})

        # Configure our logger. We do not use __name__ here because that would give us a "fibad.fibad" logger
        # which would not aggregate logs from fibad.downloadCutout which creates its own logger
        # for backwards compatibility with its CLI.
        #
        # Choosing "fibad" as our log name also means that modules like fibad.models, or fibad.dataloaders
        # can get a logger with logging.getLogger(__name__) and will automatically roll up to us.
        #
        # A downside is that multiple Fibad objects all log the same place and have the ability to clobber
        # one another's settings, and combine logs with one another.
        self.logger = logging.getLogger("fibad")

        # default to warning level
        level = general_config.get("log_level", "warning")
        if not isinstance(level, str):
            self.logger.setLevel(logging.WARNING)
        else:
            # In python 3.11 and above you can do this:
            #
            # level_map = logging.getLevelNamesMapping()
            #
            # but for now we do this:
            level_map = {
                "critical": logging.CRITICAL,
                "error": logging.ERROR,
                "warning": logging.WARNING,
                "warn": logging.WARNING,
                "info": logging.INFO,
                "debug": logging.DEBUG,
            }
            level = level_map.get(level.lower(), logging.WARNING)
            self.logger.setLevel(level)

        # Default to stderr for destination
        log_destination = general_config.get("log_destination", "stderr")

        if not isinstance(log_destination, str):
            # Default to stderr in cases where configured log_destination has a non string object
            handler = logging.StreamHandler(stream=sys.stderr)
        else:
            if log_destination.lower() == "stdout":
                handler = logging.StreamHandler(stream=sys.stdout)
            elif log_destination.lower() == "stderr":
                handler = logging.StreamHandler(stream=sys.stderr)
            else:
                handler = logging.FileHandler(log_destination)

        self.logger.addHandler(handler)

        # Format our log messages
        formatter = logging.Formatter("[%(asctime)s %(name)s:%(levelname)s] %(message)s")
        handler.setFormatter(formatter)

        self.logger.info(f"Runtime Config read from: {config_file}")

    def train(self, **kwargs):
        """
        See Fibad.train.run()
        """
        from .train import run

        return run(config=self.config, **kwargs)

    def download(self, **kwargs):
        """
        See Fibad.download.run()
        """
        from .download import run

        return run(config=self.config, **kwargs)

    def predict(self, **kwargs):
        """
        See Fibad.predict.run()
        """
        from .predict import run

        return run(config=self.config, **kwargs)

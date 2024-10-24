import logging
import sys
from pathlib import Path
from typing import Union

from .config_utils import ConfigManager, resolve_runtime_config


class Fibad:
    """
    Overall class that represents an interface into fibad. Currently this encapsulates a configuration and is
    the external interface to all verbs in a programmatic context.

    CLI functions in fibad_cli are implemented by calling this class
    """

    verbs = ["train", "predict", "download"]

    def __init__(self, *, config_file: Union[Path, str] = None, setup_logging: bool = True):
        """Initialize fibad. Always applies the default config, and merges it with any provided config file.

        Parameters
        ----------
        config_file : Union[Path, str], optional
            filename or pathlib.Path to a config file, by default None

        setup_logging : bool, optional
            Logging setup for a fibad object is global loggers named "fibad.*" If you want to turn off
            logging config for "fibad.*" python loggers, pass False here. By default True.

            You may want to set this to True if:
            - You have multiple Fibad objects in your application or notebook, and would like to control
              which of their logging configs is used globally. By creating one of your objects with
              setup_logging=True and the others with setup_logging=False, the single object created with
              setup_logging=True will control where the log is emitted to and what the threshold level is.
            - You have another library which needs overall control over python logging's config, and you
              do not want fibad to alter any global logging config. In this case you should always pass
              setup_logging=False. Fibad will stil send logs into python logging; however, the other
              system will be responsible for where those logs are emitted, and what the threshold level
              is.

            You may want to leave the default of setup_logging=True if:
            - You have a single Fibad object in use at any time. This is true in most notebook like
              environments.
        """
        self.config_manager = ConfigManager(runtime_config_filepath=config_file)
        self.config = self.config_manager.config

        # Configure our logger. We do not use __name__ here because that would give us a "fibad.fibad" logger
        # which would not aggregate logs from fibad.downloadCutout which creates its own
        # "fibad.downloadCutout" logger. It uses this name to preserve backwards compatibility with its CLI.
        #
        # Choosing "fibad" as our log name also means that modules like fibad.models, or fibad.dataloaders
        # can get a logger with logging.getLogger(__name__) and will automatically roll up to us.
        #
        self.logger = logging.getLogger("fibad")

        # The logger object we get from logging.getLogger() is a global process-level object.
        #
        # This creates some tension around what fibad initialization ought happen when the log handlers for
        # fibad have already been configured by another object!
        #
        # There are two imagined situations at time of writing where this tension occurs:
        #
        # 1) A notebook user has a cell with `fibad_instance = fibad.Fibad()` and they run this multiple times
        #    in the same kernel process. If this person is changing any logging in their fibad config file
        #    and rerunning (perhaps to debug an issue) they likely expect the most recently __init__()ed
        #    fibad object to control their logging.
        #
        # 2) An application (or notebook) user configures multiple Fibad instance objects intending to use
        #    several different configs. If the logging configs differ between those objects we don't have a
        #    clear choice about what the global fibad logging config ought be, unless the user tells us
        #    somehow.
        #
        # In the face of this tension we allow a kwarg to be passed (setup_logging) which defaults to True
        #
        # In the default case setup_logging=True, User 1 gets their expected behavior where the most recently
        # initialized object controls global logging.
        #
        # User 2 can initialize some (or all!) of their objects with setup_logging = False to achieve
        # appropriate logging behavior for their application.
        #
        if setup_logging:
            # Remove all prior handlers
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)

            # Setup our handlers from config
            self._initialize_log_handlers()

        self.logger.info(f"Runtime Config read from: {resolve_runtime_config(config_file)}")

    def _initialize_log_handlers(self):
        """Private initialization helper, Adds handlers and level setting to the global self.logger object"""

        general_config = self.config["general"]
        # default to warning level
        level = general_config["log_level"]
        if not isinstance(level, str):
            level = logging.WARNING
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
        log_destination = general_config["log_destination"]

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

    def raw_data_dimensions(self) -> tuple[list[int], list[int]]:
        """Gives the dimensions of underlying data that forms input to the training, and inference
        steps. This is the raw data that the data loader must normalize to the model

        Returns
        -------
        tuple[list[int],list[int]]
            widths and heights of all images available locally.
        """
        from .download import Downloader

        downloader = Downloader(config=self.config)
        manifest = downloader.get_manifest()
        widths = [int(dim[0]) for dim in manifest["dim"]]
        heights = [int(dim[1]) for dim in manifest["dim"]]
        return widths, heights

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
        from .download import Downloader

        downloader = Downloader(config=self.config)
        return downloader.run(**kwargs)

    def predict(self, **kwargs):
        """
        See Fibad.predict.run()
        """
        from .predict import run

        return run(config=self.config, **kwargs)

from pathlib import Path
from typing import Union

from .config_utils import get_runtime_config


class Fibad:
    """
    Overall class that represents an interface into fibad. Currently this encapsulates a configuration and is
    the external interface to all verbs in a programmatic context.

    CLI functions in fibad_cli are implemented by calling this class
    """

    #! We could potentially make this dynamic
    #! Somewhat difficult (perhaps impossible) to get this list from importlib/pkglib given
    #! That a fibad verb is simply an object in a file.
    verbs = ["train", "predict", "download"]

    def __init__(self, *, config_file: Union[Path, str] = None):
        """Initialize fibad. Always applies the default config, and merges it with any provided config file.

        Parameters
        ----------
        config_file : Union[Path, str], optional
            filename or pathlib.Path to a config file, by default None
        """
        self.config = get_runtime_config(runtime_config_filepath=config_file)

        # TODO Logging gets set up here

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

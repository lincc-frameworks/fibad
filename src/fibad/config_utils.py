import datetime
import importlib
import logging
from pathlib import Path
from typing import Union

import toml

DEFAULT_CONFIG_FILEPATH = Path(__file__).parent.resolve() / "fibad_default_config.toml"
DEFAULT_USER_CONFIG_FILEPATH = Path.cwd() / "fibad_config.toml"

logger = logging.getLogger(__name__)


class ConfigDict(dict):
    """The purpose of this class is to ensure key errors on config dictionaries return something helpful.
    and to discourage mutation actions on config dictionaries that should not happen at runtime.
    """

    # TODO: Should there be some sort of "bake" method which occurs after config processing, and
    # percolates down to nested ConfigDicts and prevents __setitem__ and other mutations of dictionary
    # values? i.e. a method to make a config dictionary fully immutable (or very difficult/annoying to
    # mutuate) before we pass control to possibly external module code that is relying on the dictionary
    # to be static througout the run.

    __slots__ = ()  # we don't need __dict__ on this object at all.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace all dictionary keys with values recursively.
        for key, val in self.items():
            if isinstance(val, dict) and not isinstance(val, ConfigDict):
                self[key] = ConfigDict(val)

    def __missing__(self, key):
        msg = f"Accessed configuration key/section {key} which has not been defined. "
        msg += "All configuration keys and sections must be defined in {DEFAULT_CONFIG_FILEPATH}"
        logger.fatal(msg)
        raise RuntimeError(msg)

    def get(self, key, default=None):
        """Nonfunctional stub of dict.get() which errors always"""
        msg = f"ConfigDict.get({key},{default}) called. "
        msg += "Please index config dictionaries with [] or __getitem__() only. "
        msg += "Configuration keys and sections must be defined in {DEFAULT_CONFIG_FILEPATH}"
        logger.fatal(msg)
        raise RuntimeError(msg)

    def __delitem__(self, key):
        raise RuntimeError("Removing keys or sections from a ConfigDict using del is not supported")

    def pop(self, key, default):
        """Nonfunctional stub of dict.pop() which errors always"""
        raise RuntimeError("Removing keys or sections from a ConfigDict using pop() is not supported")

    def popitem(self):
        """Nonfunctional stub of dict.popitem() which errors always"""
        raise RuntimeError("Removing keys or sections from a ConfigDict using popitem() is not supported")

    def clear(self):
        """Nonfunctional stub of dict.clear() which errors always"""
        raise RuntimeError("Removing keys or sections from a ConfigDict using clear() is not supported")


class ConfigManager:
    """A class to manage the runtime configuration for a Fibad object. This class
    will contain all the logic and methods for reading, merging, and validating
    the runtime configuration.
    """

    def __init__(
        self,
        runtime_config_filepath: Union[Path, str] = None,
        default_config_filepath: Union[Path, str] = DEFAULT_CONFIG_FILEPATH,
    ):
        self.fibad_default_config = self._read_runtime_config(default_config_filepath)

        self.runtime_config_filepath = runtime_config_filepath
        if self.runtime_config_filepath is None:
            self.user_specific_config = ConfigDict()
        else:
            self.user_specific_config = self._read_runtime_config(self.runtime_config_filepath)

        self.external_library_config_paths = self._find_external_library_default_config_paths(
            self.user_specific_config
        )

        self.overall_default_config = {}
        self._merge_defaults()

        self.config = self.merge_configs(self.overall_default_config, self.user_specific_config)
        self._validate_runtime_config(self.config, self.overall_default_config)

    @staticmethod
    def _read_runtime_config(config_filepath: Union[Path, str] = DEFAULT_CONFIG_FILEPATH) -> ConfigDict:
        """Read a single toml file and return a ConfigDict

        Parameters
        ----------
        config_filepath : Union[Path, str], optional
            The path to the config file, by default DEFAULT_CONFIG_FILEPATH

        Returns
        -------
        ConfigDict
            The contents of the toml file as a ConfigDict
        """
        config_filepath = Path(config_filepath)
        parsed_dict = {}
        if config_filepath.exists():
            with open(config_filepath, "r") as f:
                parsed_dict = toml.load(f)

        return ConfigDict(parsed_dict)

    @staticmethod
    def _find_external_library_default_config_paths(runtime_config: dict) -> set:
        """Search for external libraries in the runtime configuration and gather the
        libpath specifications so that we can load the default configs for the libraries.

        Parameters
        ----------
        runtime_config : dict
            The runtime configuration.
        Returns
        -------
        set
            A tuple containing the default configuration Paths for the external
            libraries that are requested in the users configuration file.
        """

        default_configs = set()
        for key, value in runtime_config.items():
            if isinstance(value, dict):
                default_configs |= ConfigManager._find_external_library_default_config_paths(value)
            else:
                if key == "name" and "." in value:
                    external_library = value.split(".")[0]
                    if importlib.util.find_spec(external_library) is not None:
                        try:
                            lib = importlib.import_module(external_library)
                            lib_default_config_path = Path(lib.__file__).parent / "default_config.toml"
                            if lib_default_config_path.exists():
                                default_configs.add(lib_default_config_path)
                        except ModuleNotFoundError:
                            logger.error(
                                f"External library {lib} not found. Please install it before running."
                            )
                            raise

        return default_configs

    def _merge_defaults(self):
        """Merge the default configurations from the fibad and external libraries."""

        # Merge all external library default configurations first
        for path in self.external_library_config_paths:
            external_library_config = self._read_runtime_config(path)
            self.overall_default_config = self.merge_configs(
                self.overall_default_config, external_library_config
            )

        # Merge the external library default configurations with the fibad default configuration
        self.overall_default_config = self.merge_configs(
            self.fibad_default_config, self.overall_default_config
        )

    @staticmethod
    def merge_configs(default_config: dict, overriding_config: dict) -> dict:
        """Merge two configurations dictionaries with the overriding_config values
        overriding the default_config values.

        Parameters
        ----------
        default_config : dict
            The default configuration.
        overriding_config : dict
            The user defined configuration.

        Returns
        -------
        dict
            The merged configuration.
        """

        final_config = default_config.copy()
        for k, v in overriding_config.items():
            if k in final_config and isinstance(final_config[k], dict) and isinstance(v, dict):
                final_config[k] = ConfigManager.merge_configs(default_config[k], v)
            else:
                final_config[k] = v

        return final_config

    @staticmethod
    def _validate_runtime_config(runtime_config: ConfigDict, default_config: ConfigDict):
        """Recursive helper to check that all keys in runtime_config have a default
        in the merged default_config.

        The two arguments passed in must represent the same nesting level of the
        runtime config and all default config parameters respectively.

        Parameters
        ----------
        runtime_config : ConfigDict
            Nested config dictionary representing the runtime config.
        default_config : ConfigDict
            Nested config dictionary representing the defaults

        Raises
        ------
        RuntimeError
            Raised if any config that exists in the runtime config does not have a default defined in
            default_config
        """
        for key in runtime_config:
            if key not in default_config:
                msg = f"Runtime config contains key or section {key} which has no default defined. "
                msg += f"All configuration keys and sections must be defined in {DEFAULT_CONFIG_FILEPATH}"
                raise RuntimeError(msg)

            if isinstance(runtime_config[key], dict):
                if not isinstance(default_config[key], dict):
                    msg = (
                        f"Runtime config contains a section named {key} which is the name of a value in the "
                    )
                    msg += "default config. Please choose another name for this section."
                    raise RuntimeError(msg)
                ConfigManager._validate_runtime_config(runtime_config[key], default_config[key])


def resolve_runtime_config(runtime_config_filepath: Union[Path, str, None] = None) -> Path:
    """Resolve a user-supplied runtime config to where we will actually pull config from.

    1) If a runtime config file is specified, we will use that file
    2) If no file is specified and there is a file named "fibad_config.toml" in the cwd we will use that file
    3) If no file is specified and there is no file named "fibad_config.toml" in the current working directory
       we will exclusively work off the configuration defaults in the packaged "fibad_default_config.toml"
       file.

    Parameters
    ----------
    runtime_config_filepath : Union[Path, str, None], optional
        Location of the supplied config file, by default None

    Returns
    -------
    Path
        Path to the configuration file ultimately used for config resolution. When we fall back to the
        package supplied default config file, the Path to that file is returned.
    """
    if isinstance(runtime_config_filepath, str):
        runtime_config_filepath = Path(runtime_config_filepath)

    # If a named config exists in cwd, and no config specified on cmdline, use cwd.
    if runtime_config_filepath is None and DEFAULT_USER_CONFIG_FILEPATH.exists():
        runtime_config_filepath = DEFAULT_USER_CONFIG_FILEPATH

    if runtime_config_filepath is None:
        runtime_config_filepath = DEFAULT_CONFIG_FILEPATH

    return runtime_config_filepath


def create_results_dir(config: ConfigDict, postfix: Union[Path, str]) -> Path:
    """Creates a results directory for this run.

    Postfix is the verb name of the run e.g. (predict, train, etc)

    The directory is created within the results dir (set with config results_dir)
    and follows the pattern <timestamp>-<postfix>

    The resulting directory is returned.

    Parameters
    ----------
    config : ConfigDict
        The full runtime configuration for this run
    postfix : str
        The verb name of the run.

    Returns
    -------
    Path
        The path created by this function
    """
    results_root = Path(config["general"]["results_dir"]).resolve()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = results_root / f"{timestamp}-{postfix}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def log_runtime_config(runtime_config: dict, output_path: Path, file_name: str = "runtime_config.toml"):
    """Log a runtime configuration.

    Parameters
    ----------
    runtime_config : dict
        A dictionary containing runtime configuration values.
    output_path : str
        The path to put the config file
    file_name : str, Optional
        Optional name for the config file, defaults to "runtime_config.toml"
    """
    with open(output_path / file_name, "w") as f:
        f.write(toml.dumps(runtime_config))

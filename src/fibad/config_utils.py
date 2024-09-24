import datetime
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

    def __init__(self, map: dict, **kwargs):
        super().__init__(map, **kwargs)

        # Replace all dictionary keys with values recursively.
        for key in self:
            if isinstance(self[key], dict) and not isinstance(self[key], ConfigDict):
                self[key] = ConfigDict(map=self[key])

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


def validate_runtime_config(runtime_config: ConfigDict):
    """Validates that defaults exist for every config value before we begin to use a config.

    This should be called at the moment the runtime config is fully baked for science calculations. Meaning
    that all sources of config info have been combined in `runtime_config` and there are no further
    config altering operations that will be performed.

    Parameters
    ----------
    runtime_config : ConfigDict
        The current runtime config dictionary.

    Raises
    ------
    RuntimeError
        Raised if any config that exists in the runtime config does not have a default defined
    """
    default_config = _read_runtime_config(DEFAULT_CONFIG_FILEPATH)
    _validate_runtime_config(runtime_config, default_config)


def _validate_runtime_config(runtime_config: ConfigDict, default_config: ConfigDict):
    """Recursive helper for validate_runtime_config.

    The two arguments passed in must represent the same nesting level of the runtime config and all
    default config parameters respectively.

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
            msg = f"Runtime config contains key or section {key} which has no default defined."
            msg += f"All configuration keys and sections must be defined in {DEFAULT_CONFIG_FILEPATH}"
            raise RuntimeError(msg)

        if isinstance(runtime_config[key], dict):
            if not isinstance(default_config[key], dict):
                msg = f"Runtime config contains a section named {key} which is the name of a value in the "
                msg += "default config. Please choose another name for this section."
                raise RuntimeError(msg)
            _validate_runtime_config(runtime_config[key], default_config[key])


def _read_runtime_config(config_filepath: Union[Path, str] = DEFAULT_CONFIG_FILEPATH) -> ConfigDict:
    """Read a single toml file and return a config dictionary

    Parameters
    ----------
    config_filepath : Union[Path, str], optional
        What file is to be read, by default DEFAULT_CONFIG_FILEPATH

    Returns
    -------
    ConfigDict
        The contents of that toml file as nested ConfigDicts
    """
    with open(config_filepath, "r") as f:
        parsed_dict = toml.load(f)
        return ConfigDict(parsed_dict)


def resolve_runtime_config(runtime_config_filepath: Union[Path, str, None] = None) -> Path:
    """Resolve a user-supplied runtime config to where we will actually pull config from.

    1) If a runtime config file is specified, we will use that file
    2) If not file is specified and there is a file named "fibad_config.toml" in the cwd we will use that file
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


def get_runtime_config(
    runtime_config_filepath: Union[Path, str, None] = None,
    default_config_filepath: Union[Path, str] = DEFAULT_CONFIG_FILEPATH,
) -> dict:
    """This function will load the default runtime configuration file, as well
    as the user defined runtime configuration file.

    The two configurations will be merged with values in the user defined config
    overriding the values of the default configuration.

    The final merged config will be returned as a dictionary and saved as a file
    in the results directory.

    Parameters
    ----------
    runtime_config_filepath : Union[Path, str, None]
        The path to the runtime configuration file.
    default_config_filepath : Union[Path, str]
        The path to the default runtime configuration file.

    Returns
    -------
    dict
        The parsed runtime configuration.
    """

    runtime_config_filepath = resolve_runtime_config(runtime_config_filepath)
    default_runtime_config = _read_runtime_config(default_config_filepath)

    if runtime_config_filepath is not DEFAULT_CONFIG_FILEPATH:
        if not runtime_config_filepath.exists():
            raise FileNotFoundError(f"Runtime configuration file not found: {runtime_config_filepath}")
        users_runtime_config = _read_runtime_config(runtime_config_filepath)
        final_runtime_config = merge_configs(default_runtime_config, users_runtime_config)
    else:
        final_runtime_config = default_runtime_config

    return final_runtime_config


def merge_configs(default_config: dict, user_config: dict) -> dict:
    """Merge two configurations dictionaries with the user_config values overriding
    the default_config values.

    Parameters
    ----------
    default_config : dict
        The default configuration.
    user_config : dict
        The user defined configuration.

    Returns
    -------
    dict
        The merged configuration.
    """

    final_config = default_config.copy()
    for k, v in user_config.items():
        if k in final_config and isinstance(final_config[k], dict) and isinstance(v, dict):
            final_config[k] = merge_configs(default_config[k], v)
        else:
            final_config[k] = v

    return final_config


def create_results_dir(config: ConfigDict, prefix: Union[Path, str]) -> Path:
    """Creates a results directory for this run.

    Prefix is the verb name of the run e.g. (predict, train, etc)

    The directory is created within the results dir (set with config results_dir)
    and follows the pattern <prefix>-<timestamp>

    The resulting directory is returned.

    Parameters
    ----------
    config : ConfigDict
        The full runtime configuration for this run
    prefix : str
        The verb name of the run.

    Returns
    -------
    Path
        The path created by this function
    """
    results_root = Path(config["general"]["results_dir"]).resolve()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = results_root / f"{prefix}-{timestamp}"
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

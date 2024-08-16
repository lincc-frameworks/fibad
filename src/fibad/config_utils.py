from pathlib import Path
from typing import Union

import toml

DEFAULT_CONFIG_FILEPATH = Path(__file__).parent.resolve() / "fibad_default_config.toml"


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

    if isinstance(runtime_config_filepath, str):
        runtime_config_filepath = Path(runtime_config_filepath)

    with open(default_config_filepath, "r") as f:
        default_runtime_config = toml.load(f)

    if runtime_config_filepath is not None:
        if not runtime_config_filepath.exists():
            raise FileNotFoundError(f"Runtime configuration file not found: {runtime_config_filepath}")

        with open(runtime_config_filepath, "r") as f:
            users_runtime_config = toml.load(f)

            final_runtime_config = merge_configs(default_runtime_config, users_runtime_config)
    else:
        final_runtime_config = default_runtime_config

    # ~ Uncomment when we have a better place to stash results.
    # log_runtime_config(final_runtime_config)

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
            final_config[k] = merge_configs(default_config.get(k, {}), v)
        else:
            final_config[k] = v

    return final_config


def log_runtime_config(runtime_config: dict, output_filepath: str = "runtime_config.toml"):
    """Log a runtime configuration.

    Parameters
    ----------
    runtime_config : dict
        A dictionary containing runtime configuration values.
    output_filepath : str
        The path to the output configuration file
    """

    with open(output_filepath, "w") as f:
        f.write(toml.dumps(runtime_config))

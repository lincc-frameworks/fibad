import os

from fibad.config_utils import ConfigManager


def test_merge_configs():
    """Basic test to ensure that the merge_configs function will join two dictionaries
    correctly, meaning:
    1) The user_config values should override the default_config values.
    2) Values in the default_config that are not in the user_config should remain unchanged.
    3) Values in the user_config that are not in the default_config should be added.
    4) Nested dictionaries should be merged recursively.
    """
    default_config = {
        "a": 1,
        "b": 2,  # This tests case 2
        "c": {"d": 3, "e": 4},
    }

    user_config = {
        "a": 5,  # This tests case 1
        "c": {
            "d": 6  # This tests case 4
        },
        "f": 7,  # This tests case 3
    }

    expected = {"a": 5, "b": 2, "c": {"d": 6, "e": 4}, "f": 7}

    assert ConfigManager.merge_configs(default_config, user_config) == expected


def test_get_runtime_config():
    """Test that the get_runtime_config function will load the default and user defined
    runtime configuration files, merge them, and return the final configuration as a
    dictionary.
    """

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_manager = ConfigManager(
        runtime_config_filepath=os.path.abspath(
            os.path.join(this_file_dir, "./test_data/test_user_config.toml")
        ),
        default_config_filepath=os.path.abspath(
            os.path.join(this_file_dir, "./test_data/test_default_config.toml")
        ),
    )

    runtime_config = config_manager.config

    expected = {
        "general": {"use_gpu": False},
        "train": {
            "model_name": "example_model",
            "model_class": "new_thing.cool_model.CoolModel",
            "model": {"model_weights_filepath": "final_best.pth", "layers": 3},
        },
        "predict": {"batch_size": 8},
    }

    assert runtime_config == expected

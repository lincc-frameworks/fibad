import pytest

from hyrax.pytorch_ignite import create_splits


def mkconfig(train_size=0.2, test_size=0.6, validate_size=0.1, seed=False):
    """Makes a configuration that has enough keys for create_splits"""
    return {
        "data_set": {
            "seed": seed,
            "train_size": train_size,
            "test_size": test_size,
            "validate_size": validate_size,
        },
    }


def test_split():
    """Test splitting in the default config where train, test, and validate are all specified"""

    fake_dataset = [1] * 100
    config = mkconfig()
    indexes = create_splits(fake_dataset, config)

    assert len(indexes["validate"]) == 10
    assert len(indexes["test"]) == 60
    assert len(indexes["train"]) == 20


def test_split_no_validate():
    """Test splitting when validate is overridden"""
    fake_dataset = [1] * 100
    config = mkconfig(validate_size=False)
    indexes = create_splits(fake_dataset, config)

    assert len(indexes["test"]) == 60
    assert len(indexes["train"]) == 20
    assert indexes.get("validate") is None


def test_split_with_validate_no_test():
    """Test splitting when validate is provided by test size is not"""
    fake_dataset = [1] * 100
    config = mkconfig(test_size=False, validate_size=0.2)
    indexes = create_splits(fake_dataset, config)

    assert len(indexes["validate"]) == 20
    assert len(indexes["test"]) == 60
    assert len(indexes["train"]) == 20


def test_split_with_validate_no_test_no_train():
    """Test splitting when validate is provided by test size is not"""
    fake_dataset = [1] * 100
    config = mkconfig(test_size=False, train_size=False, validate_size=0.2)
    indexes = create_splits(fake_dataset, config)

    assert len(indexes["test"]) == 55
    assert len(indexes["train"]) == 25
    assert len(indexes["validate"]) == 20


def test_split_with_validate_with_test_no_train():
    """Test splitting when validate is provided by test size is not"""
    fake_dataset = [1] * 100
    config = mkconfig(test_size=0.6, train_size=False, validate_size=0.2)
    indexes = create_splits(fake_dataset, config)

    assert len(indexes["test"]) == 60
    assert len(indexes["train"]) == 20
    assert len(indexes["validate"]) == 20


def test_split_no_validate_no_test():
    """Test splitting when validate and test are overridden"""
    fake_dataset = [1] * 100
    config = mkconfig(validate_size=False, test_size=False)
    indexes = create_splits(fake_dataset, config)

    assert len(indexes["test"]) == 80
    assert len(indexes["train"]) == 20
    assert indexes.get("validate") is None


def test_split_no_validate_no_train():
    """Test splitting when validate and train are overridden"""
    fake_dataset = [1] * 100
    config = mkconfig(validate_size=False, train_size=False)
    indexes = create_splits(fake_dataset, config)

    assert len(indexes["test"]) == 60
    assert len(indexes["train"]) == 40
    assert indexes.get("validate") is None


def test_split_invalid_ratio():
    """Test that split RuntimeErrors when provided with an invalid ratio"""
    fake_dataset = [1] * 100

    with pytest.raises(RuntimeError):
        create_splits(fake_dataset, mkconfig(validate_size=False, train_size=1.1))

    with pytest.raises(RuntimeError):
        create_splits(fake_dataset, mkconfig(validate_size=False, train_size=-0.1))


def test_split_no_splits_configured():
    """Test splitting when all splits are overriden, and nothing is specified."""
    fake_dataset = [1] * 100
    config = mkconfig(validate_size=False, test_size=False, train_size=False)
    indexes = create_splits(fake_dataset, config)

    assert len(indexes["test"]) == 75
    assert len(indexes["train"]) == 25
    assert indexes.get("validate") is None


def test_split_values_configured():
    """Test splitting when all splits are integer data counts"""

    fake_dataset = [1] * 100
    config = mkconfig(validate_size=22, test_size=56, train_size=22)
    indexes = create_splits(fake_dataset, config)

    assert len(indexes["test"]) == 56
    assert len(indexes["train"]) == 22
    assert len(indexes["validate"]) == 22


def test_split_values_configured_no_validate():
    """Test splitting when all splits are integer data counts and validate is not configured
    so the total selected data doesn't cover the dataset.
    """
    fake_dataset = [1] * 100
    config = mkconfig(test_size=56, train_size=22)
    indexes = create_splits(fake_dataset, config)

    assert len(indexes["test"]) == 56
    assert len(indexes["train"]) == 22
    assert len(indexes["validate"]) == 10


def test_split_invalid_configured():
    """Test that split RuntimeErrors when provided with an invalid datapoint count"""
    fake_dataset = [1] * 100

    with pytest.raises(RuntimeError):
        create_splits(fake_dataset, mkconfig(validate_size=False, train_size=120))

    with pytest.raises(RuntimeError):
        create_splits(fake_dataset, mkconfig(validate_size=False, train_size=-10))


def test_split_values_rng():
    """Generate twice with the same RNG seed, verify same values are selected."""
    fake_dataset = [1] * 100
    config = mkconfig(test_size=56, train_size=22, seed=5)
    indexes_a = create_splits(fake_dataset, config)
    indexes_b = create_splits(fake_dataset, config)

    assert all([a == b for a, b in zip(indexes_a, indexes_b)])

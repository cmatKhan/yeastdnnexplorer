import numpy as np
import pandas as pd
import pytest

from yeastdnnexplorer.ml_models.stratification_classification import (
    stratification_classification,
)


@pytest.fixture
def test_data():
    binding = pd.Series([0.1, 2.5, 1.2, 3.0, 0.8])
    perturbation = pd.Series([1.1, 0.4, 2.2, 3.0, 0.5])
    return binding, perturbation


def test_binding_only_labels(test_data):
    binding, perturbation = test_data
    labels = stratification_classification(
        binding,
        perturbation,
        bins=[0, 2, 4, np.inf],
        bin_by_binding_only=False,
    )
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(binding)
    assert all(1 <= label <= 4 for label in labels)


def test_combined_stratification_labels(test_data):
    binding, perturbation = test_data
    labels = stratification_classification(
        binding, perturbation, bins=[0, 2, 4, np.inf]
    )
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(binding)
    # With 4 bins, we expect (binding_bin - 1) * 4 + perturbation_bin to
    # be between 1 and 16
    assert all(1 <= label <= 16 for label in labels)


def test_invalid_length_raises():
    binding = pd.Series([0.1, 0.2])
    perturbation = pd.Series([0.1])
    with pytest.raises(
        ValueError, match="length of the binding and perturbation vectors must be equal"
    ):
        stratification_classification(binding, perturbation)


def test_non_series_input_raises():
    binding = [0.1, 0.2]
    perturbation = pd.Series([0.1, 0.2])
    with pytest.raises(ValueError, match="binding vector must be a pandas Series"):
        stratification_classification(binding, perturbation)


def test_non_numeric_series_raises():
    binding = pd.Series(["a", "b", "c"])
    perturbation = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="binding vector must be numeric"):
        stratification_classification(binding, perturbation)


def test_too_few_bins_returns_ones(test_data, caplog):
    binding, perturbation = test_data
    with caplog.at_level("WARNING"):
        result = stratification_classification(binding, perturbation, bins=[0])
    assert all(result == 1)
    assert "The number of bins is less than 2" in caplog.text

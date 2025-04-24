import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("main")


def stratification_classification(
    binding_series: pd.Series,
    perturbation_series: pd.Series,
    bins: list = [0, 8, 64, 512, np.inf],
    bin_by_binding_only: bool = False,
) -> np.ndarray:
    """
    Bin the binding and perturbation data and create groups for stratified k folds.

    :param binding_series: The binding vector to use for stratification
    :param perturbation_series: The perturbation vector to use for stratification
    :param bins: The bins to use for stratification.
        The default is [0, 8, 64, 512, np.inf]
    :param bin_by_binding_and_perturbation: If False (default) use both the binding and
        perturbation data to create a combined stratification label. This will
        create a unique label for each combination of binding and perturbation bins.
        Setting this to True will bin by the binding data only.

    :return: A numpy array of the stratified classes

    :raises ValueError: If the length of `binding_series` and
        `perturbation_series` are not equal (when binding_only=False)
    :raises ValueError: If the length of `bins` is less than 2
    :raises ValueError: If `binding_series` and `perturbation_series` are not numeric
        pd.Series

    """
    # validate the binding data
    if not isinstance(binding_series, pd.Series):
        raise ValueError("The binding vector must be a pandas Series.")
    if binding_series.dtype.kind not in "biufc":
        raise ValueError("The binding vector must be numeric.")
    # validate the perturbation data
    if not isinstance(perturbation_series, pd.Series):
        raise ValueError("The perturbation vector must be a pandas Series.")
    if perturbation_series.dtype.kind not in "biufc":
        raise ValueError("The perturbation vector must be numeric.")
    # validate that the binding and perturbation series are are the same length
    if len(binding_series) != len(perturbation_series):
        raise ValueError(
            "The length of the binding and perturbation vectors must be equal."
        )

    # if the number of bins is less than 2, return a pd.Series of ones
    if len(bins) < 2:
        logger.warning(
            "The number of bins is less than 2. Returning a series of ones."
            "This will not provide meaningful stratification."
        )
        return np.ones(len(binding_series), dtype=int)

    # create labels for the bins, eg 1, 2, 3, 4 if the bins are [0, 8, 64, 512, np.inf]
    labels = list(range(1, len(bins)))

    # bin the binding data by ranking, and then using the bins and labels to
    # assign rows to bins
    binding_rank = binding_series.rank(method="min", ascending=False).values
    binding_bin = pd.cut(binding_rank, bins=bins, labels=labels, right=True).astype(int)

    # if bin_by_binding_and_perturbation is False, return the binding_bin
    if not bin_by_binding_only:
        return binding_bin

    # if binding_only is False, bin the perturbation data by ranking, and then using
    # the bins and labels to assign rows to bins
    perturbation_rank = perturbation_series.rank(method="min", ascending=False).values
    perturbation_bin = pd.cut(
        perturbation_rank, bins=bins, labels=labels, right=True
    ).astype(int)

    # Transform the binding_bin by multiplying it by a scale factor
    # (the number of labels). This ensures that combining
    # (binding_bin - 1) * len(labels) + perturbation_bin
    # produces a unique label for each (binding_bin, perturbation_bin) pair.
    # For example, if binding_bin = 2 and perturbation_bin = 3, and there are 4 labels,
    # then the final label will be (2 - 1) * 4 + 3 = 7.
    # This allows for stratification across both variables simultaneously.
    return (binding_bin - 1) * len(labels) + perturbation_bin

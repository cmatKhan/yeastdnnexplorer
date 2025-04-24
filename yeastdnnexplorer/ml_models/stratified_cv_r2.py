import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def stratified_cv_r2(
    y: pd.DataFrame,
    X: pd.DataFrame,
    classes: np.ndarray,
    estimator: BaseEstimator = LinearRegression(fit_intercept=True),
    skf: StratifiedKFold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42),
    **kwargs,
) -> float:
    """
    Calculate the average stratified CV r-squared for a given estimator and data. By
    default, this is a 4-fold stratified CV with a LinearRegression estimator. Note that
    by default, the estimator is set to LinearRegression() and the StratifiedKFold
    object is set to a 4-fold stratified CV with shuffle=True and random_state=42.
    LinearRegression has fit_intercept explicitly set to True, meaning the data IS NOT
    expected to be centered and there should not be a constant column in X.

    :param y: The response variable. See generate_modeling_data()
    :param X: The predictor variables. See generate_modeling_data()
    :param classes: the stratification classes for the data
    :param estimator: the estimator to be used in the modeling. By default, this is a
        LinearRegression() model.
    :param skf: the StratifiedKFold object to be used in the modeling. By default, this
        is a 4-fold stratified CV with shuffle=True and random_state=42.
    :return: the average r-squared value for the stratified CV

    """
    estimator_local = clone(estimator)
    r2_scores = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        folds = list(skf.split(X, classes))
        for warning in w:
            logger.debug(
                f"Warning encountered during stratified k-fold split: {warning.message}"
            )

    for train_idx, test_idx in folds:
        # Use train and test indices to split X and y
        X_train, X_test = (
            X.iloc[train_idx],
            X.iloc[test_idx],
        )
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit the model
        model = estimator_local.fit(
            X_train,
            y_train,
        )

        # Calculate R-squared and append to r2_scores
        r2_scores.append(r2_score(y_test, model.predict(X_test)))

    return np.mean(r2_scores)

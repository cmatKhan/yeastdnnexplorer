import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from yeastdnnexplorer.ml_models.SigmoidModel import SigmoidModel, sigmoid
from yeastdnnexplorer.ml_models.stratification_classification import (
    stratification_classification,
)


@pytest.fixture
def setup_glm_two_variable():
    """Fixture to set up the GeneralizedLogisticModel with sample data."""

    # Generate sample data
    np.random.seed(42)
    X1 = np.linspace(-10, 10, 100)
    X2 = np.linspace(-10, 10, 100)
    X0 = np.ones_like(X1)

    # True sigmoid parameters
    true_left_asymptote = 1.3
    true_right_asymptote = 9.5
    true_slope = np.array([-7, 1.6, 1.6])  # Slopes for both variables

    # Stack X1 and X2 to form a design matrix with two variables
    X_two_vars = np.column_stack([X0, X1, X2])

    # Generate Y data using the true sigmoid function
    Y_true = sigmoid(true_left_asymptote, true_right_asymptote, X_two_vars @ true_slope)

    # Add some noise to the Y values
    noise = 0.75 * np.random.randn(len(X1))
    Y_noisy = Y_true.ravel() + noise

    classes = stratification_classification(pd.Series(X1), pd.Series(Y_noisy))

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    folds = list(skf.split(X1, classes))

    glm = SigmoidModel(cv=folds, alphas=[10, 11, 12])

    return glm, X_two_vars, Y_noisy


def test_weighted_sum_squares():
    # simple test to check that multiplying two np.array of same length mulitply
    # element wise
    actual = np.array([4, 4, 4, 4]) * np.array([0.5, 0.25, 1.0, 0.75])
    expected = np.array([2.0, 1.0, 4.0, 3.0])
    np.allclose(actual, expected, atol=1e-17)

    # Example data
    y = np.array([2.0, 1.5, 3.0, 4.0, 5.0])
    pred = np.array([1.8, 1.7, 2.5, 4.5, 4.8])
    sample_weight = np.array([1.0, 0.5, 2.0, 1.5, 0.1])

    # Method 1: sqrt weights, then square residuals
    residual_1 = (y - pred) * np.sqrt(sample_weight)
    sse_1 = np.sum(residual_1**2)

    # Method 2: square residuals, then apply weights
    residual_2 = y - pred
    sse_2 = np.sum(sample_weight * residual_2**2)

    sse_1, sse_2, np.allclose(sse_1, sse_2, atol=1e-17)


def test_model_fitting(setup_glm_two_variable):
    glm, X_two_vars, Y_noisy = setup_glm_two_variable

    # Fit the model to the noisy data
    glm.fit(X_two_vars, Y_noisy)

    # Check if the coefficients have been fitted
    assert glm.coef_ is not None, "Coefficients were not fitted."
    assert glm.left_asymptote_ is not None, "Left asymptote was not fitted."
    assert glm.right_asymptote_ is not None, "Right asymptote was not fitted."

    # Check if the fitted asymptotes are close to the true values
    assert np.isclose(
        glm.left_asymptote_, 1.3, atol=0.5
    ), "Left asymptote is incorrect."
    assert np.isclose(
        glm.right_asymptote_, 9.5, atol=0.5
    ), "Right asymptote is incorrect."


def test_model_prediction(setup_glm_two_variable):
    glm, X_two_vars, Y_noisy = setup_glm_two_variable

    # Fit the model and predict
    glm.fit(X_two_vars, Y_noisy)
    Y_fitted = glm.predict(X_two_vars)

    # Ensure the predicted values are close to the noisy data
    assert Y_fitted.shape == Y_noisy.shape, "Predicted values shape mismatch."
    assert np.isclose(
        np.mean(Y_fitted), np.mean(Y_noisy), atol=0.5
    ), "Prediction error is too large."


def test_r_squared(setup_glm_two_variable):
    glm, X_two_vars, Y_noisy = setup_glm_two_variable

    # Fit the model and calculate R-squared
    glm.fit(X_two_vars, Y_noisy)

    # Check if R-squared is a valid value
    r_squared = glm.score(X_two_vars, Y_noisy)
    assert r_squared is not None, "R-squared was not calculated."
    assert 0 <= r_squared <= 1, "R-squared is out of bounds."


# def test_summary(setup_glm_two_variable, capsys):
#     glm, X_two_vars, Y_noisy = setup_glm_two_variable

#     # Fit the model and generate the summary
#     glm.fit(X_two_vars, Y_noisy)
#     glm.summary()

#     # Capture the output of the summary
#     captured = capsys.readouterr()

#     # Check if the summary contains expected sections
#     assert "Likelihood Ratio Test" in captured.out, "LRT output is missing."


# def test_plot_execution(setup_glm_two_variable):
#     glm, X_two_vars, Y_noisy = setup_glm_two_variable

#     # Fit the model and run the plot method
#     glm.fit(X_two_vars, Y_noisy)

#     # Just test if the plot runs without errors
#     glm.plot(interactor_diagnostic=False)

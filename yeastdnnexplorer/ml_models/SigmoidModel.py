import logging
from collections.abc import Sequence

import numpy as np
from numpy.random import default_rng
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

logger = logging.getLogger("main")


class LinearModelSampler:
    """Draws coefficient samples from a multivariate normal distribution centered at the
    OLS estimate with covariance based on residual variance and XtX^{-1}."""

    def __init__(self, X, y, sample_weight=None):
        self.X = X
        self.y = y
        self.sample_weight = sample_weight
        self._fit_model()

    def _fit_model(self):
        lr = LinearRegression()
        lr.fit(self.X, self.y, sample_weight=self.sample_weight)
        self.beta_hat_ = lr.coef_
        self.intercept_ = lr.intercept_

        y_pred = lr.predict(self.X)
        residuals = self.y - y_pred
        n, p = self.X.shape
        dof = n - p
        self.sigma_squared_ = np.sum(residuals**2) / dof

        XtX = self.X.T @ self.X
        self.cov_beta_ = self.sigma_squared_ * np.linalg.pinv(XtX)

    def sample(self, random_state=None):
        rng = default_rng(random_state)
        return rng.multivariate_normal(mean=self.beta_hat_, cov=self.cov_beta_)


def sigmoid(left_asymptote, right_asymptote, linear_combination):
    """
    Y(X) = (right_asymptote - left_asymptote) * sigmoid(X @ B) + left_asymptote
    """
    amplitude = right_asymptote - left_asymptote
    return amplitude * expit(linear_combination) + left_asymptote


def objective(w, X, y, alpha, sample_weight=None):
    """
    Objective function: weighted SSE + L1 penalty on coefficients.

    The sigmoid portion is:
        Y(X) = (right_asymptote - left_asymptote) * sigmoid(X @ B) + left_asymptote

    And the L1 penalty is the sum of the coefficients, excluding the asymptotes. Note
        that we do include B_0
    """
    left_asymptote = w[0]
    right_asymptote = w[1]
    linear_combination = X @ w[2:]
    pred = sigmoid(left_asymptote, right_asymptote, linear_combination)
    residual = y - pred

    if sample_weight is not None:
        # mulitplying numpy arrays of the same length multiply element wise. See
        # test_sigmoid_modeling for verification
        try:
            sse = np.sum(sample_weight * residual**2)
        except ValueError as exc:
            raise ValueError("sample_weight must be the same length as y.") from exc
    else:
        sse = np.sum(residual**2)

    penalty = alpha * np.sum(np.abs(w[2:]))
    return sse + penalty


class SigmoidModel(BaseEstimator, RegressorMixin):
    """Scikit-learn-compatible estimator for generalized logistic regression with L1
    penalty."""

    def __init__(
        self,
        random_state: int = 42,
        warm_start: bool = True,
        cv: list[tuple] | None = None,
        alpha: float | None = None,
        alphas: Sequence[float] | None = None,
    ):
        self.cv = cv
        self.alpha = alpha
        self.alphas = alphas
        self.random_state = random_state
        self.warm_start = warm_start

    def fit(
        self,
        X,
        y,
        method="L-BFGS-B",
        minimize_options=None,
        sample_weight=None,
    ):
        if minimize_options is None:
            minimize_options = {}
        else:
            logger.info("Using minimize_options: %s", minimize_options)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            if len(sample_weight) != len(y):
                raise ValueError("sample_weight must be the same length as y.")

        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = getattr(X, "columns", None)

        if self.warm_start:
            logger.info("warm start is True. Creating linear model coef sampler")
            lms = LinearModelSampler(X, y, sample_weight=sample_weight)

        # Validate cv
        if self.cv is None or not (
            isinstance(self.cv, list)
            and all(isinstance(fold, tuple) and len(fold) == 2 for fold in self.cv)
        ):
            raise ValueError("cv must be a list of (train_idx, test_idx) tuples.")

        # Validate alphas
        if self.alphas is not None:
            self.alphas_ = np.asarray(self.alphas, dtype=float)

        # Validate alpha
        if self.alpha is not None:
            try:
                self.alpha_ = float(self.alpha)
            except ValueError:
                raise ValueError("alpha must be a castable to float.")

        # initialize parameters to 0s
        init_params = np.zeros(X.shape[1] + 2)
        # set right asymptote init value to 1
        init_params[1] = 1.0
        # if warm_start is true, then sample a coefficient value from the distribution
        # of each coefficient from a linear fit
        if self.warm_start:
            init_params[2:] = lms.sample(random_state=self.random_state)

        self.init_params_ = init_params

        # Either use fixed alpha, set above, or use cross validation to select and set
        # an alpha_
        if not hasattr(self, "alpha_"):
            # If either alpha or alphas is present in the instance, raise an
            # AttributeError
            if not hasattr(self, "alphas_"):
                raise AttributeError("Either `alpha` or `alphas` must be provided.")

            mse_path = np.zeros((len(self.alphas_), len(self.cv)))
            for a_idx, alpha in enumerate(self.alphas_):
                for f_idx, (train_idx, test_idx) in enumerate(self.cv):
                    X_train, y_train = X[train_idx], y[train_idx]
                    X_test, y_test = X[test_idx], y[test_idx]
                    weights_train = (
                        sample_weight[train_idx] if sample_weight is not None else None
                    )

                    fold_result = minimize(
                        objective,
                        self.init_params_.copy(),
                        args=(X_train, y_train, alpha, weights_train),
                        method=method,
                        options=minimize_options,
                    )

                    # calculate test prediction
                    left_asymptote = fold_result.x[0]
                    right_asymptote = fold_result.x[1]
                    linear_combination = X_test @ fold_result.x[2:]
                    pred = sigmoid(left_asymptote, right_asymptote, linear_combination)

                    # calculate test residuals
                    residual = y_test - pred

                    # based on wether sample_weight is None or not, calculate the mse
                    if sample_weight is not None:
                        mse_path[a_idx, f_idx] = np.average(
                            residual**2, weights=sample_weight[test_idx]
                        )
                    else:
                        mse_path[a_idx, f_idx] = np.mean(residual**2)

            self.mse_path_ = mse_path
            best_alpha_idx = np.argmin(np.mean(self.mse_path_, axis=1))
            self.alpha_ = self.alphas_[best_alpha_idx]

        # Final fit on all data
        final_result = minimize(
            objective,
            self.init_params_.copy(),
            args=(X, y, self.alpha_, sample_weight),
            method=method,
            options=minimize_options,
        )

        self.left_asymptote_ = final_result.x[0]
        self.right_asymptote_ = final_result.x[1]
        self.coef_ = final_result.x[2:]

        return self

    def predict(self, X):
        check_is_fitted(
            self, attributes=["left_asymptote_", "right_asymptote_", "coef_"]
        )
        X = check_array(X)
        linear_combination = X @ self.coef_
        return sigmoid(self.left_asymptote_, self.right_asymptote_, linear_combination)

    def score(self, X, y):
        return r2_score(y, self.predict(X))

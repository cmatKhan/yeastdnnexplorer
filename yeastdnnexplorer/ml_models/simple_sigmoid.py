import logging
import numbers
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
from tabulate import tabulate  # type: ignore

from yeastdnnexplorer.utils.InteractorDiagnosticPlot import InteractorDiagnosticPlot
from yeastdnnexplorer.utils.sigmoid import sigmoid

logger = logging.getLogger("main")


class SimpleSigmoidModel:
    """Generalized logistic model for fitting sigmoidal curves to data."""

    def __init__(self, max_iter=10000, eps=1e-12):
        """Initialize the generalized logistic model."""
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._right_asymptote: float | None = None
        self._left_asymptote: float | None = None
        self._coef_: np.ndarray | None = None
        self._cov: np.ndarray | None = None
        self._residuals: np.ndarray | None = None
        self._jacobian: np.ndarray | None = None
        self.max_iter = max_iter
        self.eps = eps
        # Store initialization parameters for cloning
        self._init_params = {
            "max_iter": max_iter,
            "eps": eps,
        }

    def __copy__(self):
        """
        Create a shallow copy of the model.

        :return: A shallow copy of the model

        """
        # Create a new instance with the same initialization parameters
        clone = type(self)()

        # Update initialization parameters
        clone._init_params = self._init_params.copy()

        # Update all attributes
        clone.__dict__.update(self.__dict__)
        return clone

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the model.

        :param memo: Memo dictionary used by the deepcopy process
        :return: A deep copy of the model

        """
        import copy

        # Create a new instance with default parameters first
        clone = type(self)()

        # Add the new object to the memo dictionary
        memo[id(self)] = clone

        # Deep copy initialization parameters
        clone._init_params = copy.deepcopy(self._init_params, memo)

        # Deep copy all other attributes
        for k, v in self.__dict__.items():
            if k != "_init_params":
                setattr(clone, k, copy.deepcopy(v, memo))

        return clone

    @property
    def X(self) -> np.ndarray | None:
        """
        Set the predictor variables for the model.

        :param value: The input data matrix. Must be two dimensional even if there is
            only one predictor.
        :return: The input data matrix.
        :raises TypeError: if X is not a NumPy array.
        :raises ValueError: if X is not 2D.
        :raises ValueError: if the number of columns in X does not match the length of
            the inflection point or coefficients.

        """
        return self._X

    @X.setter
    def X(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("X must be a NumPy array.")

        # Ensure that X is 2D
        if value.ndim != 2:
            raise ValueError("X must be a 2-dimensional array.")

        if not np.all(value[:, 0] == 1):
            raise ValueError("The first column of X must be a constant vector of ones.")

        # Check compatibility with coefficients
        if self._coef_ is not None and len(self._coef_) != value.shape[1]:
            raise ValueError(
                "Coefficients must have the same number of elements as columns in X."
            )

        self._X = value

    @property
    def y(self) -> np.ndarray | None:
        """
        Set the response variable for the model.

        :param value: The observed output data.
        :return: The observed output data.
        :raises TypeError: if y is not a NumPy array or a list.
        :raises ValueError: if the number of rows in y does not match the number of rows
            in X.

        """
        return self._y

    @y.setter
    def y(self, value: np.ndarray) -> None:
        if not isinstance(value, (np.ndarray, list)):
            raise TypeError("y must be a NumPy array or a list.")

        value = np.asarray(value)

        if self.X is not None and value.shape[0] != self.X.shape[0]:
            raise ValueError("y must have the same number of rows as X.")

        self._y = value

    @property
    def right_asymptote(self) -> float | None:
        """
        Set the upper asymptote for the model. This parameter can be inferred by `fit()`

        :param value: The upper asymptote of the sigmoid function.
        :return: The upper asymptote of the sigmoid function.
        :raises TypeError: if the upper asymptote is not a real number.
        :raises ValueError: if the upper asymptote is less than the lower asymptote.

        """
        return self._right_asymptote

    @right_asymptote.setter
    def right_asymptote(self, value: float) -> None:
        if not isinstance(value, numbers.Real):
            raise TypeError("Upper asymptote must be a real number.")

        self._right_asymptote = value

    @property
    def left_asymptote(self) -> float | None:
        """
        The lower asymptote of the sigmoid function. This parameter can be inferred by
        `fit()`

        :return: The lower asymptote of the sigmoid function.
        :raises TypeError: if the lower asymptote is not a real number.
        :raises ValueError: if the lower asymptote is greater than the upper asymptote.

        """
        return self._left_asymptote

    @left_asymptote.setter
    def left_asymptote(self, value: float) -> None:
        if not isinstance(value, numbers.Real):
            raise TypeError("Lower asymptote must be a real number.")

        self._left_asymptote = value

    @property
    def coef_(self) -> np.ndarray | None:
        """
        Set the coefficients for the model. This parameter can be inferred by `fit()`

        :param value: The coefficients of the sigmoid function.
        :return: The coefficients of the sigmoid function.
        :raises TypeError: if the coefficients are not a NumPy array or a list.
        :raises ValueError: if the length of the coefficients does not match the number
            of columns in X or the number of inflection points.

        """
        return self._coef_

    @coef_.setter
    def coef_(self, value: Sequence[float] | np.ndarray) -> None:
        # validate type
        if not isinstance(value, (np.ndarray, list)):
            raise TypeError("Coefficients must be a NumPy array or a list.")

        value = np.asarray(value)

        if self.X is not None and len(value) != self.X.shape[1]:
            raise ValueError(
                "Coefficients must have the same number of elements as columns in X."
            )

        self._coef_ = value

    @property
    def cov(self) -> np.ndarray | None:
        """
        The covariance matrix of the model parameters. This parameter can be inferred by
        `fit()`

        :return: The covariance matrix of the model parameters.
        :raises TypeError: if the covariance matrix is not a NumPy array.

        """
        return self._cov

    @cov.setter
    def cov(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("cov must be a NumPy array.")
        self._cov = value

    @property
    def residuals(self) -> np.ndarray | None:
        """
        The residuals of the model. This parameter can be inferred by `fit()`

        :return: The residuals of the model.
        :raises TypeError: if the residuals are not a NumPy array.

        """
        return self._residuals

    @residuals.setter
    def residuals(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("residuals must be a NumPy array.")
        self._residuals = value

    @property
    def jacobian(self) -> np.ndarray | None:
        """
        The Jacobian matrix of the model. This parameter can be inferred by `fit()`

        :return: The Jacobian matrix of the model.
        :raises TypeError: if the Jacobian matrix is not a NumPy array.

        """
        return self._jacobian

    @jacobian.setter
    def jacobian(self, value: np.ndarray) -> None:
        # if not isinstance(value, np.ndarray):
        #     raise TypeError("Jacobian must be a NumPy array.")
        self._jacobian = value

    @property
    def n_params(self) -> int:
        """
        The number of parameters in the model.

        :return: The number of parameters in the model.
        :raises AttributeError: if the coefficients are not available.

        """
        if self.X is None:
            raise AttributeError("X not available. Set with `model()`.")
        # Number of parameters = number of coefficients + 2 (for the two asymptotes)
        return self.X.shape[1] + 2

    @property
    def df(self) -> int:
        """
        The residual degrees of freedom of the model.

        Residual degrees of freedom = number of observations - number of parameters

        :return: The residual degrees of freedom of the model.

        :raises AssertionError: if the input data matrix X is not available.

        """
        assert self.X is not None, "Input data matrix X is not available."
        # Number of parameters = number of coefficients + 2 (for the two asymptotes)
        return self.X.shape[0] - self.n_params

    @property
    def mse(self) -> float | None:
        """
        The mean squared error of the model.

        :return: The mean squared error of the model.
        :raises AttributeError: if the residuals are not available.

        """
        if self.residuals is None:
            raise AttributeError("Residuals are not available.")
        return np.mean(self.residuals**2)

    @property
    def rss(self) -> float | None:
        """
        The residual sum of squares of the model.

        :return: The residual sum of squares of the model.
        :raises AttributeError: if the residuals are not available.

        """
        if self.residuals is None:
            raise AttributeError("Residuals are not available.")
        return np.sum(self.residuals**2)

    @property
    def tss(self) -> float | None:
        """
        The total sum of squares of the model.

        :return: The total sum of squares of the model.
        :raises AttributeError: if the output data y is not available.

        """
        if self.y is None:
            raise AttributeError("Output data y is not available.")
        return np.sum((self.y - np.mean(self.y)) ** 2)

    @property
    def r_squared(self) -> float | None:
        """
        The variance explained by the model.

        :return: The variance explained by the model.

        :raises AttributeError: `rss` or `tss` is not available

        """
        if self.rss is None:
            raise AttributeError(
                "`rss` is not available. Check that `fit()` has been run."
            )
        if self.tss is None:
            raise AttributeError(
                "`tss` is not available. Check that `model()` has been run."
            )
        return 1 - (self.rss / self.tss)

    @property
    def adj_r_squared(self) -> float | None:
        """
        The adjusted variance explained by the model.

        :return: The adjusted variance explained by the model.

        :raises AttributeError: `rss` or `tss` is not available

        """
        if self.rss is None:
            raise AttributeError(
                "`rss` is not available. Check that `fit()` has been run."
            )
        if self.tss is None:
            raise AttributeError(
                "`tss` is not available. Check that `model()` has been run."
            )
        if self.df is None:
            raise AttributeError(
                "`df` is not available. Check that `fit()` has been run."
            )
        if self.X is None:
            raise AttributeError(
                "`X.shape[0]` is not available. Check that `fit()` has been run."
            )

        r2 = 1 - (self.rss / self.tss)

        return 1 - (1 - r2) * (self.X.shape[0] - 1) / self.df

    @property
    def llf(self) -> float | None:
        """
        The log-likelihood of the model. Note that this assumes Gaussian residuals.

        :return: The log-likelihood of the model.
        :raises AttributeError: if the residuals or y are not available.

        """
        if self.residuals is None:
            raise AttributeError("Residuals are not available.")
        # Number of observations
        if self.y is None:
            raise AttributeError("Output data y is not available.")
        n = len(self.y)

        # Variance of the residuals (sigma^2). denominator is N, not N-1. ddof=1 is
        # for the sample variance
        # see https://numpy.org/doc/stable/reference/generated/numpy.var.html
        sigma_squared = np.var(self.residuals, ddof=0)

        if sigma_squared <= 0:
            raise ValueError("Variance of residuals is zero or negative.")

        # Sum of squared residuals
        sum_squared_residuals = np.sum(self.residuals**2)

        # Log-likelihood using Gaussian assumption (standard OLS log likelihood)
        log_likelihood = (
            -(n / 2) * np.log(2 * np.pi * sigma_squared)
            - (1 / (2 * sigma_squared)) * sum_squared_residuals
        )

        return log_likelihood

    def aic(self) -> float | None:
        """
        Calculate the Akaike Information Criterion (AIC) for the model.

        :return: The Akaike Information Criterion (AIC) for the model.
        :raises AttributeError: if the log-likelihood is not available.

        """
        if self.llf is None:
            raise AttributeError("Log-likelihood is not available.")
        # AIC = 2p − 2log(L)
        return 2 * self.n_params - 2 * self.llf

    def bic(self) -> float | None:
        """
        Calculate the Bayesian Information Criterion (BIC) for the model.

        :return: The Bayesian Information Criterion (BIC) for the model.
        :raises AttributeError: if the log-likelihood or X is not available.

        """
        if self.llf is None:
            raise AttributeError("Log-likelihood is not available.")
        if self.X is None:
            raise AttributeError("Input data matrix X is not available.")
        # BIC = plog(n) − 2log(L)
        return self.n_params * np.log(self.X.shape[0]) - 2 * self.llf

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the generalized logistic model.

        :param X: Input data matrix
        :return: Predictions based on the learned model parameters
        :raises ValueError: if the model has not been fitted.

        """
        if self.right_asymptote is None or self.left_asymptote is None:
            raise ValueError("Model must be fitted before making predictions.")

        assert self.coef_ is not None, "Coefficients are not available."

        return sigmoid(
            X,
            self.left_asymptote,
            self.right_asymptote,
            self.coef_,
        )

    def fit(self, X, y, sample_weight=None, **kwargs):
        """
        Fit Generalized Logistic Model using MSE objective.

        :param X: Input data matrix
        :param y: Target values
        :param sample_weight: Optional sample weights (not used)
        :param kwargs: Additional arguments passed to minimize
        :return: self

        """
        self._X = np.asarray(X)
        self._y = np.asarray(y).ravel()

        def objective(w, X_local, y_local):
            """MSE objective function w[0] = left_asymptote w[1] = right_asymptote w[2:]
            = coefficients."""
            linear_combination = np.dot(X_local, w[2:])
            left_asymptote = w[0]
            right_asymptote = w[1]
            pred = left_asymptote + (right_asymptote - left_asymptote) / (
                1 + np.exp(-linear_combination)
            )
            residual = y_local - pred
            return np.sum(residual**2)  # MSE

        # Initialize parameters
        w0 = np.zeros(self._X.shape[1] + 2)
        w0[1] = 1.0

        # Run optimization
        res = minimize(
            fun=objective,
            x0=w0,
            args=(self._X, self._y),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.eps, "gtol": self.eps},
            **kwargs,
        )

        # Extract final parameters
        w_best = res.x
        self._left_asymptote = w_best[0]
        self._right_asymptote = w_best[1]
        self.coef_ = w_best[2:]
        # Compute residuals on full data
        linear_combination = np.dot(self._X, self.coef_)
        left_asymptote = (
            self._left_asymptote if self._left_asymptote is not None else 0.0
        )
        right_asymptote = (
            self._right_asymptote if self._right_asymptote is not None else 1.0
        )
        pred_full = left_asymptote + (right_asymptote - left_asymptote) / (
            1 + np.exp(-linear_combination)
        )

        self._residuals = self._y - pred_full

        # Store optimization result
        self.optimization_result_ = res

        return self

    def summary(self) -> None:
        """
        Print a summary of the generalized logistic model.

        This method automatically performs LRT comparisons between the full model and
        models with one less predictor in each iteration.

        :raises ValueError: if the model has not been fitted.

        """
        if self.X is None or self.y is None or self.coef_ is None:
            raise ValueError("Model must be fitted before generating a summary.")

        # Calculate the log-likelihood for the full model (self)
        assert self.llf is not None, "Log-likelihood not available."
        self_log_likelihood = self.llf

        print(f"self_log_likelihood: {self_log_likelihood}")

        # Fit the OLS model for comparison
        ols_model = sm.OLS(self.y, self.X)
        ols_results = ols_model.fit()
        log_likelihood_linear = ols_results.llf

        print(f"log_likelihood_linear: {log_likelihood_linear}")

        # LRT comparing the full sigmoid model to the linear model
        lrt_statistic_linear = -2 * (log_likelihood_linear - self_log_likelihood)
        p_value_linear = 1 - stats.chi2.cdf(lrt_statistic_linear, df=2)

        # Prepare the summary table for the sigmoid model
        param_names = ["left_asymptote", "right_asymptote"] + [
            f"coef_{i}" for i in range(len(self.coef_))
        ]
        estimates = np.concatenate(
            [[self.left_asymptote, self.right_asymptote], self.coef_]
        )

        # Display the model summary in a formatted table
        table_data = [
            [param, f"{est:12.4f}"] for param, est in zip(param_names, estimates)
        ]
        print("\nGeneralized Logistic Model Summary\n")
        print(tabulate(table_data, headers=["Parameter", "Estimate"], tablefmt="pipe"))

        # Improved model diagnostics table
        diagnostics_table = [
            ["Metric", "Sigmoid Model", "Linear Model"],
            [
                "Variance Explained (R-squared)",
                f"{self.r_squared:.4f}",
                f"{ols_results.rsquared:.4f}",
            ],
            [
                "Akaike Information Criterion (AIC)",
                f"{self.aic():.4f}",
                f"{ols_results.aic:.4f}",
            ],
            [
                "Bayesian Information Criterion (BIC)",
                f"{self.bic():.4f}",
                f"{ols_results.bic:.4f}",
            ],
        ]

        print("\nModel Diagnostics Comparison")
        print(tabulate(diagnostics_table, headers="firstrow", tablefmt="pipe"))

        # LRT vs linear model
        lrt_table = [
            ["Linear Model Log-Likelihood", f"{log_likelihood_linear:.4f}"],
            ["Sigmoid Model Log-Likelihood", f"{self_log_likelihood:.4f}"],
            ["LRT Statistic", f"{lrt_statistic_linear:.4f}"],
            ["p-value", f"{p_value_linear:.4g}"],
        ]
        print("\nLikelihood Ratio Test (LRT) vs Linear Model")
        print(tabulate(lrt_table, tablefmt="pipe"))

        # Iterate over columns of X, removing one column at a time and performing LRT
        lrt_comparisons = []
        # Iterate over columns of X, removing columns from the end to the beginning
        for i in range(self.X.shape[1] - 1, 0, -1):
            X_reduced = self.X[
                :, :i
            ]  # Take the first i columns (this removes the last column first)

            # Fit a reduced model with fewer columns
            reduced_model = SimpleSigmoidModel()
            reduced_model.fit(X_reduced, self.y)  # Fixed: Swapped X and y arguments

            # Calculate log-likelihood for the reduced model
            assert reduced_model.llf is not None, "Log-likelihood not available."
            log_likelihood_reduced = reduced_model.llf

            # Perform LRT between the full model and the reduced model
            lrt_statistic_reduced = -2 * (log_likelihood_reduced - self_log_likelihood)
            param_diff = 1  # Since we're removing one predictor at a time
            p_value_reduced = 1 - stats.chi2.cdf(lrt_statistic_reduced, df=param_diff)

            # Collect the results for the reduced model
            lrt_comparisons.append(
                [
                    f"Reduced Model (with first {i} columns)",
                    f"{log_likelihood_reduced:.4f}",
                    f"{lrt_statistic_reduced:.4f}",
                    f"{p_value_reduced:.4g}",
                ]
            )

        # Output the results for the reduced models
        print("\nLRT Comparisons with Reduced Models")
        print(
            tabulate(
                lrt_comparisons,
                headers=["Model", "Log-Likelihood", "LRT Statistic", "p-value"],
                tablefmt="pipe",
            )
        )

    def plot(
        self,
        plots_to_display: list[int] = [1, 2, 3, 4],
        interactor_diagnostic: bool = False,
    ):
        """
        Diagnostic plots for the generalized logistic model.

        This function can generate various plots including:

        - 1: Normal Q-Q plot.
        - Optionally: Interactor Diagnostic Plot when `interactor_diagnostic` is True.

        :param plots_to_display: A list of integers (1 to 4) indicating which
            plots to show.
        :param interactor_diagnostic: Boolean to include interactor diagnostic
            plot (default False).

        """
        if self.X is None or self.y is None or self.residuals is None:
            raise ValueError("Model must be fitted before plotting diagnostics.")

        # Create subplots grid based on number of plots to display
        n_plots = len(plots_to_display) + (1 if interactor_diagnostic else 0)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

        if n_plots == 1:
            axes = [axes]  # Ensure axes is always iterable

        for i, plot_num in enumerate(plots_to_display):
            if plot_num == 1:
                # 1. Normal Q-Q Plot
                stats.probplot(self.residuals, dist="norm", plot=axes[i])
                axes[i].set_title("Normal Q-Q")

        # If the user requested the interactor diagnostic plot
        if interactor_diagnostic:
            assert self.coef_ is not None, "Input data matrix X is not available."
            # Call the interactor diagnostic plot class
            interactor_plotter = InteractorDiagnosticPlot(
                df=pd.DataFrame(self.X),  # Your data goes here
                quantile=0.1,  # Use an appropriate quantile value
                B=self.coef_,  # Pass the model coefficients
                model_type="sigmoid",
                left_asymptote=self.left_asymptote,
                right_asymptote=self.right_asymptote,
            )
            plt_obj = interactor_plotter.plot()  # Create the diagnostic plot
            plt_obj.show()  # Show the plot in a separate figure

        plt.tight_layout()
        plt.show()

    def get_params(self, deep=True):
        """
        Return the parameters of the model.

        :param deep: Whether to return deep parameters (ignored for simplicity).
        :return: A dictionary of parameters.

        """
        return {
            "eps": self.eps,
            "max_iter": self.max_iter,
        }

    def set_params(self, **params):
        """
        Set the parameters of the model.

        :param params: A dictionary of parameters to set.
        :return: self

        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

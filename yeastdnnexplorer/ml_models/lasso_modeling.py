import logging
import re
import warnings
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy as pt
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from statsmodels.api import OLS, add_constant

logger = logging.getLogger("main")


def generate_modeling_data(
    colname: str,
    response_df: pd.DataFrame,
    predictors_df: pd.DataFrame,
    drop_intercept: bool = True,
    formula: str | None = None,
    quantile_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate the response and predictor data, optionally filtering to the top x
    quantile.

    :param colname: The column name to use as the response variable. This column name
        should exist in `response_df` and `predictors_df`.
    :param response_df: The transformed response variable DataFrame.
    :param predictors_df: The predictors DataFrame.
    :param drop_intercept: Whether to drop the intercept in the formula. This adds
        a -1 to the formula if True. See the patsy docs
        https://patsy.readthedocs.io/en/latest/formulas.html#intercept-handling.
    :param formula: The formula to use for the interaction model. If None, the formula
        will be generated automatically. The formula should be in the form of
        `colname_LRR ~ predictor1 + predictor2 + ... + predictorN`. If `drop_intercept`
        is True, a -1 will be added to the formula. Note that _LRR needs to be added
        to the response variable name in the formula.
    :param quantile_threshold: If specified, filters the data to only include rows
        where `predictors_df[colname]` is in the top `quantile_threshold` quantile.
        For example, `quantile_threshold=0.1` would include only the top 10%.

    :return: A tuple of the response variable DataFrame and the predictors DataFrame.

    :raises ValueError: If `colname` does not exist in `predictors_df`
        or `response_df`.
    :raises ValueError: If any columns in `response_df` are not present
        in `predictors_df`.

    """
    # Validate input
    if colname not in response_df.columns:
        raise ValueError(
            f"The column {colname} does not exist in the response DataFrame."
        )

    # Ensure all columns in response_df are in predictors_df (excluding rep patterns)
    # TODO: this is a hack. We are assuming that the response column might have
    # a _rep\d+, removing it, and using it as the column name. Somehow this needs to
    # be handled in a more robust way. See the other TODOS on this
    response_columns = response_df.columns.str.replace(r"_rep\d+", "", regex=True)
    missing_cols = response_columns.difference(predictors_df.columns)
    if not missing_cols.empty:
        raise ValueError(
            "The following columns are missing from the predictors DataFrame: "
            f"{missing_cols}"
        )

    # Merge response and predictors DataFrames on their index
    tmp_df = pd.merge(
        response_df[[colname]].rename(columns={colname: f"{colname}_LRR"}),
        predictors_df,
        left_index=True,
        right_index=True,
        how="inner",
    )

    # check if colname is in the index. If it is, drop it
    if colname not in tmp_df.index:
        logger.warning(
            f"{colname} is not in the index. The perturbed gene may "
            "not be removed from the modeling data"
        )
    else:
        logger.info(
            f"Removing {colname} from the data rows (removing the perturbed TF)"
        )
        tmp_df = tmp_df.drop(colname)

    # log the number of rows in the merged dataframe
    logger.info(f"Number of rows in the merged response/predictors: {tmp_df.shape[0]}")

    # remove the _rep\d+ pattern from the response colname -- perturbation data should
    # not have replicates.
    # TODO: this needs to be handled more transparently. See the other TODOs on
    # rep\d
    perturbed_tf = re.sub(r"_rep\d+", "", colname)

    # Apply quantile filtering if quantile_threshold is specified
    if quantile_threshold is not None:
        quantile_value = tmp_df[perturbed_tf].quantile(1 - quantile_threshold)
        tmp_df = tmp_df[tmp_df[perturbed_tf] >= quantile_value]
        logger.info(
            f"Number of rows after filtering by the top "
            f"{quantile_threshold} of {perturbed_tf}: {tmp_df.shape[0]}"
        )

    # need to check if max_lrb exists in the column if the input formula includes it
    if formula and "max_lrb" in formula:
        logger.info("Adding max_lrb column to the DataFrame.")
        tmp_df["max_lrb"] = predictors_df.drop(columns=perturbed_tf).max(axis=1)

    # Step 3: Define the interaction formula
    if formula is None:
        interaction_terms = " + ".join(
            [
                f"{perturbed_tf}:{other_col}"
                for other_col in predictors_df.columns
                if other_col != perturbed_tf
            ]
        )
        formula = f"{colname}_LRR ~ {perturbed_tf} + {interaction_terms}"

    if drop_intercept:
        formula += " - 1"

    # Step 4: Generate X, y matrices with patsy
    logger.info(f"Generating modeling data with formula: {formula}")
    try:
        # Attempt to create matrices
        y, X = pt.dmatrices(formula, tmp_df, return_type="dataframe")
        # Clean up column names in `y`
        y.columns = y.columns.str.replace("_LRR", "")
    except pt.PatsyError as exc:
        # Check if it's the specific missing outcome variable error
        if "model is missing required outcome variables" in str(exc):
            logger.info("No outcome variable found in the formula")
            X = pt.dmatrix(formula, tmp_df, return_type="dataframe")
            y = pd.Series(dtype="float64")
        else:
            # Re-raise the error if it's something else
            raise

    return y, X


def stratification_classification(
    binding_series: pd.Series,
    perturbation_series: pd.Series,
    bins: list = [0, 8, 64, 512, np.inf],
) -> np.ndarray:
    """
    Bin the binding and perturbation data and create groups for stratified k folds.

    :param binding_series: The binding vector to use for stratification
    :param perturbation_series: The perturbation vector to use for stratification
    :param bins: The bins to use for stratification.
        The default is [0, 8, 64, 512, np.inf]

    :return: A numpy array of the stratified classes

    :raises ValueError: If the length of `binding_series` and
        `perturbation_series` are not equal
    :raises ValueError: If the length of `bins` is less than 2
    :raises ValueError: If `binding_series` and `perturbation_series` are not numeric
        pd.Series

    """
    # Validate input
    if len(binding_series) != len(perturbation_series):
        raise ValueError(
            "The length of the binding and perturbation vectors must be equal."
        )
    if len(bins) < 2:
        raise ValueError("The number of bins must be at least 2.")
    if not isinstance(binding_series, pd.Series) or not isinstance(
        perturbation_series, pd.Series
    ):
        raise ValueError("The binding and perturbation vectors must be pandas Series.")
    if (
        binding_series.dtype.kind not in "biufc"
        or perturbation_series.dtype.kind not in "biufc"
    ):
        raise ValueError("The binding and perturbation vectors must be numeric.")

    # Rank genes by binding and perturbation scores
    binding_rank = binding_series.rank(method="min", ascending=False).values
    perturbation_rank = perturbation_series.rank(method="min", ascending=False).values

    # Generate labels based on the number of bins
    # Number of labels is one less than the number of bin edges
    labels = list(range(1, len(bins)))

    # Bin genes based on binding and perturbation ranks
    binding_bin = pd.cut(binding_rank, bins=bins, labels=labels, right=True).astype(int)
    perturbation_bin = pd.cut(
        perturbation_rank, bins=bins, labels=labels, right=True
    ).astype(int)

    # Generate a combined classification based on the binding and perturbation bins
    # each combination of binding and perturbation bin will be a unique class
    return (binding_bin - 1) * len(labels) + perturbation_bin


def stratified_cv_modeling(
    y: pd.DataFrame,
    X: pd.DataFrame,
    classes: np.ndarray,
    estimator: BaseEstimator = LassoCV(),
    skf: StratifiedKFold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42),
    sample_weight: np.ndarray | None = None,
) -> BaseEstimator:
    """
    This conducts the LassoCV modeling. The name `stratified_cv_modeling` is a misnomer.
    There is nothing in this function that requires any specific model.

    :param y: The response variable to use for modeling. This should be a single column.
        See `generate_modeling_data()`
    :param X: The predictors to use for modeling. This should be an NxP DataFrame where
        N == len(y) and P is the number of predictors. See `generate_modeling_data()`
    :param classes: The classes to use for stratified k-fold cross-validation. This
        should be an array of integers generated by `stratification_classification()`
    :param estimator: The estimator to use for fitting the model. It must have a `cv`
        attribute that can be set with a list of StratifiedKFold splits
    :param skf: The StratifiedKFold object to use for stratified splits. Default is
        StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    :param sample_weight: The sample weights to use for fitting the model. Default is
        None, which is the default behavior for LassoCV.fit()

    :return: The LassoCV model

    :raises ValueError: if y is not a single column DataFrame
    :raises ValueError: if X is not a DataFrame with 1 or more columns, or the number
        of rows in y does not match the number of rows in X
    :raises ValueError: if classes is not a numpy array or is empty
    :raises ValueError: If the estimator does not have a `cv` attribute

    """
    # Validate data
    if not isinstance(y, pd.DataFrame):
        raise ValueError("The response variable y must be a DataFrame.")
    if y.shape[1] != 1:
        raise ValueError("The response variable y must be a single column DataFrame.")
    if not isinstance(X, pd.DataFrame):
        raise ValueError("The predictors X must be a DataFrame.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("The number of rows in X must match the number of rows in y.")
    if classes.size == 0 or not isinstance(classes, np.ndarray):
        raise ValueError("The classes must be a non-empty numpy array.")

    # Verify estimator has a `cv` attribute
    if not hasattr(estimator, "cv"):
        raise ValueError("The estimator must support a `cv` parameter.")

    # Initialize StratifiedKFold for stratified splits
    logger.debug("Generating stratified k-fold splits")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        folds = list(skf.split(X, classes))
        for warning in w:
            logger.warning(
                f"Warning encountered during stratified k-fold split: {warning.message}"
            )

    # Clone the estimator and set the `cv` attribute with predefined folds
    model = clone(estimator)
    model.cv = folds

    # Step 7: Fit the model using the custom cross-validation folds
    model.fit(
        X,
        y.values.ravel(),
        sample_weight=sample_weight,
    )

    return model


def bootstrap_stratified_cv_modeling(
    n_bootstraps: int = 1000,
    ci_percentiles: list[float] = [95.0, 99.0],
    use_sample_weight_in_cv: bool = False,
    **kwargs,
) -> tuple[dict[str, dict[str, tuple[float, float]]], pd.DataFrame, list[float]]:
    r"""
    Perform bootstrap resampling to generate confidence intervals for
    Lasso coefficients. See 6.2 in https://hastie.su.domains/StatLearnSparsity/ -- this
    is an implementation of the algorithm described in that section.

    :param n_bootstraps: The number of bootstrap samples to generate.
    :param ci_percentiles: A list of CI percentiles to calculate. Default to [95, 99]
    :param use_sample_weight_in_cv: Whether to use sample weights, calculated as the
        proportion of times a given record appears, in boostrap iterations. Default is
        False.
    :param kwargs: The required arguments to `stratified_cv_modeling` must be
        passed as keyword arguments to this function. Any additional keyword arguments
        will be passed to the Lasso estimator.

    :return: A tuple where:
        - The first element is a dictionary of confidence intervals,
          where keys are CI levels (e.g., "95.0") and values are dictionaries
          mapping each coefficient to its lower and upper bounds, with columns named
          according to the predictors in `X`.
        - The second element is a DataFrame where each row represents a
        bootstrap sample,
          and columns correspond to the predictor variables from `X`.
        - The third element is a list of alpha values from each bootstrap iteration.


    :raises ValueError: If any of the required keyword arguments for
        `stratified_cv_modeling()` are not passed.
    :raises ValueError: If `n_bootstraps` is not an integer greater than 0.
    :raises ValueError: If `ci_percentiles` is not a list of integers or floats.
    :raises ValueError: If `use_sample_weight_in_cv` is not a boolean.
    :raises ValueError: If the response variable is not in the predictors DataFrame.
        If there are replicates, they are expected to have the suffix _rep\d+. This
        is attempted to be removed from the response variable name to match the
        predictors DataFrame.
    """
    # Validate input parameters
    if not isinstance(n_bootstraps, int) or n_bootstraps < 1:
        raise ValueError("The number of bootstraps must be an integer greater than 0.")
    if not isinstance(ci_percentiles, list) or not all(
        isinstance(x, (int, float)) for x in ci_percentiles
    ):
        raise ValueError(
            "The confidence interval percentiles must be a list of integers or floats."
        )
    if not isinstance(use_sample_weight_in_cv, bool):
        raise ValueError("The use_sample_weight_in_cv parameter must be a boolean.")

    # Extract and validate required arguments for modeling
    y = kwargs.pop("y")
    X = kwargs.pop("X")
    estimator = kwargs.pop("estimator")
    if any(x is None for x in [y, X, estimator]):
        raise ValueError(
            "Arguments 'y', 'X', and 'estimator' must be "
            "passed to bootstrap_cv_modeling()."
        )

    response_tf = re.sub(r"_rep\d+", "", y.columns[0])

    if response_tf not in X.columns:
        raise ValueError(
            f"The response variable {response_tf} is not in the predictors DataFrame."
        )

    bootstrap_coefs = []
    alpha_list = []

    # Bootstrap iterations
    for _ in range(n_bootstraps):
        Y_resampled = resample(y, replace=True)
        X_resampled = X.loc[Y_resampled.index]

        weights = None
        if use_sample_weight_in_cv:
            index_mapping = {label: idx for idx, label in enumerate(y.index)}
            integer_indices = [index_mapping[label] for label in Y_resampled.index]
            sample_counts = np.bincount(integer_indices, minlength=len(y))
            weights = sample_counts / len(y)

        classes = stratification_classification(
            X_resampled.loc[:, response_tf].squeeze(), Y_resampled.squeeze()
        )

        # this is the second part of the code for the edge case in which during
        # sequential modeling we do not find the main effect to be significant
        # in the first place, therefore we must ignore it for modeling purposes
        ignore_main_effects = kwargs.get("ignore_main_effect", False)
        if ignore_main_effects:
            X_resampled = X_resampled.drop([response_tf], axis=1)

        model_i = stratified_cv_modeling(
            Y_resampled,
            X_resampled,
            classes=classes,
            estimator=estimator,
            sample_weight=weights,
        )
        alpha_list.append(model_i.alpha_)
        bootstrap_coefs.append(model_i.coef_)

    # Convert coefficients list to a DataFrame with column names from X
    bootstrap_coefs_df = pd.DataFrame(
        bootstrap_coefs,
        # this is the fourth part of the code that handles the edge case where you drop
        # the main effect if needed
        columns=(
            X.drop(columns=[response_tf]).columns if ignore_main_effects else X.columns
        ),
    )

    # Compute confidence intervals
    ci_dict = {
        f"{str(ci)}": {
            colname: (
                np.percentile(bootstrap_coefs_df[colname], (100 - ci) / 2),
                np.percentile(bootstrap_coefs_df[colname], 100 - (100 - ci) / 2),
            )
            for colname in bootstrap_coefs_df.columns
        }
        for ci in ci_percentiles
    }

    return ci_dict, bootstrap_coefs_df, alpha_list


def examine_bootstrap_coefficients(
    lasso_model_output, ci_level: str = "95.0", threshold: float = 0.0
) -> tuple[Figure, dict[str, tuple[float, float]]]:
    """
    Generate a plot, and output the set of coefficients that meet the ci_level and
    threshold criteria.

    Usage:

    Examples
    --------
    Display the plot immediately:

    ```python
    fig, significant_cols = plot_significant_lasso_coefficients(
        lasso_model_output,
        ci_level="95.0",
        threshold=0.5
    )
    fig.show()
    ```

    Further customize the plot:

    ```python
    fig, significant_cols = plot_significant_lasso_coefficients(
        lasso_model_output,
        ci_level="95.0",
        threshold=0.5
    )
    ax = fig.gca()  # Get the current axes
    ax.set_title("Custom Title for Lasso Coefficients", fontsize=16)
    ax.set_ylabel("Custom Y-axis Label")
    fig.show()
    ```

    Save the plot to a file:

    ```python
    fig, significant_cols = plot_significant_lasso_coefficients(
        lasso_model_output,
        ci_level="95.0",
        threshold=0.5
    )
    fig.savefig("significant_lasso_coefficients.png", dpi=300, bbox_inches='tight')
    ```

    Embed in Jupyter Notebooks:

    ```python
    plot_significant_lasso_coefficients(
        lasso_model_output,
        ci_level="95.0",
        threshold=0.5
    )
    ```

    Add annotations or modify appearance:

    ```python
    fig, significant_cols = plot_significant_lasso_coefficients(
        lasso_model_output,
        ci_level="95.0",
        threshold=0.5
    )
    ax = fig.gca()
    ax.annotate("Important Coefficient", xy=(1.5, 1), xytext=(2, 2),
                arrowprops=dict(facecolor='black', shrink=0.05))
    fig.show()
    ```

    :param lasso_model_output: The output from `bootstrap_cv_modeling()`.
        Expected format: (fitted_model, ci_dict, bootstrap_coefs_df, alpha_list)
    :param ci_level: The confidence interval level to use, e.g., "95.0".
    :param threshold: The threshold for selecting coefficients to plot.
        Only coefficients with confidence intervals entirely
        above or below this threshold will be displayed.

    :return: A tuple containing:
        - The created Matplotlib Figure for further customization.
        - A dictionary where the keys are the significant coefficients and the values
        are the confidence intervals specified in the `ci_level` parameter.

    """
    # Unpack lasso_model_output
    ci_dict, bootstrap_coefs_df, _ = lasso_model_output

    ci_dict_local = ci_dict.copy()

    # if ci_level is not in ci_dict, then it needs to be calculated from the
    # boostrap_coef_df
    if ci_level not in ci_dict_local:
        ci_dict_local[ci_level] = {
            colname: (
                np.percentile(bootstrap_coefs_df[colname], (100 - float(ci_level)) / 2),
                np.percentile(
                    bootstrap_coefs_df[colname], 100 - (100 - float(ci_level)) / 2
                ),
            )
            for colname in bootstrap_coefs_df.columns
        }

    # Identify coefficients with confidence intervals entirely above or
    # below the threshold, with a minimum distance from zero.
    # We always ignore the Intercept because it is not a true predictor
    significant_coefs_dict = {
        coef: bounds
        for coef, bounds in ci_dict_local[ci_level].items()
        if coef != "Intercept" and (bounds[0] > threshold or bounds[1] < -threshold)
    }

    # Display selected coefficients and their intervals
    print(
        f"Significant coefficients for {ci_level}, "
        f"where intervals are entirely above or below ±{threshold}:"
    )
    for coef, bounds in significant_coefs_dict.items():
        print(f"{coef}: {bounds}")

    # Extract the selected columns from bootstrap_coefs_df for plotting
    df_extracted = bootstrap_coefs_df[significant_coefs_dict.keys()]

    # Plotting the boxplot of the selected coefficients
    fig = plt.figure(figsize=(10, 6))

    # Add subtitle details about the threshold
    title_text = f"Coefficients with {ci_level}% CI intervals outside ±{threshold}"

    sns.boxplot(data=df_extracted, orient="h")
    plt.axvline(x=0, linestyle="--", color="black")
    plt.xlabel("Coefficient Values")

    # Main title and subtitle
    plt.title(title_text)

    return fig, significant_coefs_dict


def get_significant_predictors(
    method: Literal["lassocv_ols", "bootstrap_lassocv"],
    perturbed_tf: str,
    response_df: pd.DataFrame,
    predictors_df: pd.DataFrame,
    add_max_lrb: bool,
    **kwargs: Any,
) -> dict[str, dict | pd.DataFrame | np.ndarray | tuple]:
    """
    This function is used to get the significant predictors for a given TF using one of
    two methods, either the bootstrapped LassoCV, in which case we look for intervals
    that do not cross 0, or direct LassoCV with a selection on non-zero coefficients.

    :param method: This must be 'lassocv_ols', which will conduct a single lassocv call
        followed by pruning non zero coefficients by pvalue until all are significant
        at a given threshold, or 'bootstrap_lassocv', which will conduct bootstrapped
        lassoCV and return only coefficients which are deemed significant by
        ci_percentile and threshold (see `examine_bootstrap_coeficients` for more info)
    :param perturbed_tf: the TF for which the significant predictors are to be
        identified
    :param response_df: The DataFrame containing the response values
    :param predictors_df: The DataFrame containing the predictor values
    :param add_max_lrb: A boolean to add/not add in the max_LRB term for a response TF
        into the formula that we perform bootstrapping on
    :param kwargs: Additional arguments to be passed to the function. Expected arguments
        are 'quantile_threshold' from generate_modeling_data() and 'ci_percentile' from
        examine_bootstrap_coefficients()

    :return a dictionary with keys 'sig_coefs', 'response', 'classes', and
        'bootstrap_lasso_output' where:
        - 'sig_coefs' is a dictionary of significant coefficients
        - 'response' is the response variable
        - 'classes' is the stratification classes for the data
        - 'bootstrap_lasso_output' is the bootstrapping output for intermediate model
        analysis if the method chosen is 'bootstrap_lassocv', otherwise it will be None

    """
    if method not in ["lassocv_ols", "bootstrap_lassocv"]:
        raise ValueError(
            "method {} unrecognized. "
            "Must be one of ['lassocv_ols', 'bootstrap_lassocv']"
        )

    # this is the first part of the code that checks for the particular scenario in
    # which the main effect isn't in the formula if it is not there, then we must add it
    # when generating the modeling data, and then ignore it for the actual modeling
    # which is done by the other part of the code in this method and in
    # bootstrap_cv_modeling
    # TODO: modify generate_modeling_data to avoid the requirement of having the
    # main effect in the columns
    formula = kwargs.get("formula", None)
    if formula:
        lhs, rhs = formula.split("~")
        rhs = rhs.strip()
        rhs_terms = [term.strip() for term in rhs.split("+")]

        if perturbed_tf not in rhs_terms:
            formula = f"{lhs.strip()} ~ {rhs} + {perturbed_tf}"

    y, X = generate_modeling_data(
        perturbed_tf,
        response_df,
        predictors_df,
        quantile_threshold=kwargs.get("quantile_threshold", None),
        drop_intercept=True,
        formula=formula,  # Pass formula from kwargs
    )

    # NOTE: fit_intercept is set to `true`
    lassoCV_estimator = LassoCV(
        fit_intercept=True,
        max_iter=10000,
        selection="random",
        random_state=42,
        n_jobs=4,
    )

    predictor_variable = re.sub(r"_rep\d+", "", perturbed_tf)

    stratification_classes = stratification_classification(
        X[predictor_variable].squeeze(), y.squeeze()
    )

    if add_max_lrb:
        # add a column to X which is the rowMax excluding column `predictor_variable`
        # called max_lrb
        max_lrb = X.drop(columns=predictor_variable).max(axis=1)
        X["max_lrb"] = max_lrb

    # Fit the model to the data in order to extract the alphas_ which are generated
    # during the fitting process
    lasso_model = stratified_cv_modeling(
        y, X, stratification_classes, lassoCV_estimator
    )

    if method == "lassocv_ols":
        # return a list of the non-zero features that survived the fitting
        non_zero_indices = lasso_model.coef_ != 0
        non_zero_features = X.columns[non_zero_indices]
        sig_coef_dict = {
            k: v for k, v in zip(non_zero_features, lasso_model.coef_[non_zero_indices])
        }

    elif method == "bootstrap_lassocv":

        # set the alphas_ attribute of the lassoCV_estimator to the alphas_
        # attribute of the lasso_model fit on the whole data. This will allow the
        # bootstrap_stratified_cv_modeling function to use the same set of lambdas
        lassoCV_estimator.alphas_ = lasso_model.alphas_

        # this is the third part specifically checks an edge case in sequential modeling
        # where after the lasso model on all genes is run and the main effect isn't
        # considered significant. Thus, we cannot consider it further, but we still need
        # to use it in this method to stratify datapoints for bootstrapping. This check
        # will enure that during bootstrap_stratified_cv_modeling we ultimately don't
        # also pass in the main effect
        # NOTE: hacky, because generate_modeling_data() requires main effect.
        # See issue in git.
        formula = kwargs.get("formula", None)
        ignore_main_effect = False  # Default to False

        if formula:
            # Extract the right-hand side (RHS) of the formula (everything after `~`)
            rhs = formula.split("~")[-1].strip()

            # Tokenize by splitting on "+" and stripping spaces
            rhs_terms = [term.strip() for term in rhs.split("+")]

            # Check if `perturbed_tf` is missing from the RHS terms
            if perturbed_tf not in rhs_terms:
                ignore_main_effect = True

        logging.info("running bootstraps")
        bootstrap_lasso_output = bootstrap_stratified_cv_modeling(
            y=y,
            X=X,
            estimator=lassoCV_estimator,
            ci_percentile=kwargs.get("ci_percentile", 95.0),
            n_bootstraps=kwargs.get("n_bootstraps", 1000),
            max_iter=10000,
            fit_intercept=True,
            selection="random",
            random_state=42,
            ignore_main_effect=ignore_main_effect,
        )

        sig_coef_plt, sig_coef_dict = examine_bootstrap_coefficients(
            bootstrap_lasso_output, ci_level=kwargs.get("ci_percentile", 95.0)
        )

        plt.close(sig_coef_plt)

    else:
        ValueError(f"method {method} not recognized")

    result = {
        "sig_coefs": sig_coef_dict,
        "predictors": X,
        "response": y,
        "classes": stratification_classes,
        "bootstrap_lasso_output": (
            bootstrap_lasso_output if method == "bootstrap_lassocv" else None
        ),
    }

    return result


def stratified_cv_r2(
    y: pd.DataFrame,
    X: pd.DataFrame,
    classes: np.ndarray,
    estimator: BaseEstimator = LinearRegression(),
    skf: StratifiedKFold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42),
) -> float:
    """
    Calculate the average stratified CV r-squared for a given estimator and data. By
    default, this is a 4-fold stratified CV with a LinearRegression estimator. Note that
    this method will add an intercept to X if it doesn't already exist.

    :param y: The response variable. See generate_modeling_data()
    :param X: The predictor variables. See generate_modeling_data()
    :param classes: the stratification classes for the data
    :param estimator: the estimator to be used in the modeling. By default, this is a
        LinearRegression() model.
    :param skf: the StratifiedKFold object to be used in the modeling. By default, this
        is a 4-fold stratified CV with shuffle=True and random_state=42.
    :return: the average r-squared value for the stratified CV

    """
    # If there is no constant term, add one
    X_with_intercept = add_constant(X, has_constant="skip")

    estimator_local = clone(estimator)
    r2_scores = []

    for train_idx, test_idx in skf.split(X_with_intercept, classes):
        # Use train and test indices to split X and y
        X_train, X_test = (
            X_with_intercept.iloc[train_idx],
            X_with_intercept.iloc[test_idx],
        )
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit the model
        model = estimator_local.fit(X_train, y_train)

        # Calculate R-squared and append to r2_scores
        r2_scores.append(r2_score(y_test, model.predict(X_test)))

    return np.mean(r2_scores)


def try_interactor_variants(
    intersect_coefficients: set[str], interactor: str, **kwargs: Any
) -> list[dict[str, Any]]:
    """
    For a given interactor, replace the term in the formula with one variant:
        1. the main effect
    For this variant, calculate the average stratified CV r-squared with
    stratified_cv_r2().

    :param intersect_coefficients: the set of coefficients that are determined to be
        significant, expected to be from either a bootstrap procedure on a LassoCV
        model on a full partition of the data and the top 10% by perturbed binding, or
        LassoCV followed by backwards selection by adj-rsquared.
    :param interactor: the interactor term to be tested
    :param kwargs: additional arguments to be passed to the function. Expected
        arguments are 'y', 'X', and 'stratification_classes'. See stratified_cv_r2()
        for more information.

    :return: a list with three dict entries, each with key
        'interactor', 'variant', 'avg_r2'
    """
    y = kwargs.get("y")
    if y is None:
        raise ValueError("y must be passed as a keyword argument")
    X = kwargs.get("X")
    if X is None:
        raise ValueError("X must be passed as a keyword argument")
    stratification_classes = kwargs.get("stratification_classes")
    if stratification_classes is None:
        raise ValueError("stratification_classes must be passed as a keyword argument")

    # the main effect is assumed to be the second term in the interactor
    main_effect = interactor.split(":")[1]

    # Add different ways of replacing the interactor term here. if only [main_effect],
    # then the only variant tested will be replacing the interactor term with the main
    # effect
    interactor_formula_variants = [main_effect]

    output = []
    for variant in interactor_formula_variants:
        # replace the interactor term in the formula with the variant
        variant_predictors = (
            [term for term in intersect_coefficients if term != interactor] + [variant]
            if isinstance(variant, str)
            else variant
        )
        # conduct the stratified CV r-squared calculation with the formula variant
        input_model_avg_rsquared = stratified_cv_r2(
            y, X.loc[:, variant_predictors], stratification_classes
        )

        # append the results to the output list
        output.append(
            {
                "interactor": interactor,
                "variant": variant,
                "avg_r2": input_model_avg_rsquared,
            }
        )
    return output


def get_interactor_importance(
    y: pd.DataFrame,
    full_X: pd.DataFrame,
    stratification_classes: np.ndarray,
    intersect_coefficients: set,
) -> tuple[float, list[dict[str, Any]]]:
    """
    For each interactor in the intersect_coefficients, run test_interactor_importance to
    compare the variants' avg_rsquared to the input_model_avg_rsquared. If a variant of
    the interactor term is better, record it in a dictionary. Return the
    `instersect_coefficient` model's avg R-squared and the dictionary of interaction
    alternatives that, when that alternative replaces a single interaction term,
    improves the rsquared.

    :param y: the response variable
    :param full_X: the full predictor matrix
    :param stratification_classes: the stratification classes for the data
    :param intersect_coefficients: the set of coefficients that are determined to be
        significant, expected to be from either a bootstrap procedure on a LassoCV model
        on a full partition of the data and the top 10% by perturbed binding, or LassoCV
        followed by backwards selection by p-value significance.
    :return: a tuple with the first element being the input_model_avg_rsquared and the
        second element being a list of dictionaries with keys 'interactor', 'variant',
        and 'avg_r2'

    """

    input_model_avg_rsquared = stratified_cv_r2(
        y, full_X.loc[:, list(intersect_coefficients)], stratification_classes
    )

    # for each interactor in the intersect_coefficients, run test_interactor_importance
    # compare the variant's avg_rsquared to the input_model_avg_rsquared. Record
    # the best performing.
    interactor_results = []
    for interactor in intersect_coefficients:
        if ":" in interactor:

            interactor_variant_results = try_interactor_variants(
                intersect_coefficients,
                interactor,
                y=y,
                X=full_X,
                stratification_classes=stratification_classes,
            )

            # Find the variant with the maximum avg_r2
            best_variant = max(interactor_variant_results, key=lambda x: x["avg_r2"])

            # Compare its avg_r2 to the input model's avg_r2
            if best_variant["avg_r2"] > input_model_avg_rsquared:
                interactor_results.append(best_variant)

    return input_model_avg_rsquared, interactor_results


class OLSFeatureSelector(BaseEstimator, TransformerMixin):
    """
    This class performs iterative feature selection using OLS.

    It removes non-significant features until all remaining features are significant.

    """

    def __init__(self, p_value_threshold=0.05):
        """
        Initialize the OLSFeatureSelector.

        :param p_value_threshold: The threshold for significance of features.

        """
        self.p_value_threshold = p_value_threshold
        self.significant_features_ = []
        self.summary_ = None

    def fit(self, X, y, **kwargs) -> "OLSFeatureSelector":
        """
        Fit the OLS model and identify significant features. Significant features are
        selected based based on coef p-value <= p_value_threshold.

        :param X: A DataFrame of predictors.
        :param y: A Series of the response variable.
        :param kwargs: Optional arguments for `add_constant()`.
        :return: self

        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"X must be a DataFrame, but got {type(X)}.")

        # Add an intercept term
        X_with_intercept = add_constant(
            X, has_constant=kwargs.get("has_constant", "skip")
        )

        # Fit the OLS model
        model = OLS(y, X_with_intercept).fit()

        # Save the summary table
        summary_data = {
            "coef": model.params,
            "std_err": model.bse,
            "t": model.tvalues,
            "pvalue": model.pvalues,
        }
        self.summary_ = pd.DataFrame(summary_data)

        # Select significant features based on p-values
        self.significant_features_ = [
            feature
            for feature, pval in model.pvalues.items()
            if pval <= self.p_value_threshold and feature != "const"
        ]
        return self

    def transform(self, X) -> pd.DataFrame:
        """
        Iteratively apply OLS to remove non-significant features.

        :param X: A DataFrame of predictors.
        :return: A DataFrame with only significant features.

        """
        if not self.significant_features_:
            raise ValueError("The model has not been fitted yet. Call `fit` first.")

        # Ensure the input DataFrame contains all required columns
        missing_features = set(self.significant_features_) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        return X[self.significant_features_]

    def refine_features(self, X, y) -> pd.DataFrame:
        """
        Iteratively fit the selector and transform the data in one step.

        :param X: A DataFrame of predictors.
        :param y: A Series of the response variable.
        :return: A DataFrame with only significant predictors.

        """
        remaining_predictors = X.copy()

        while True:
            # Fit the model
            self.fit(remaining_predictors, y)

            # If all predictors are significant, stop iteration
            if set(self.significant_features_) == set(remaining_predictors.columns):
                break

            # Retain only significant predictors for the next iteration
            remaining_predictors = remaining_predictors[self.significant_features_]

        return remaining_predictors

    def get_significant_features(self, drop_intercept=True) -> list:
        """
        Get the list of significant features.

        param drop_intercept: Whether to exclude the intercept term from the list.
            NOTE: this only looks for a feature called "Intercept"

        :return: List of significant feature names.

        """
        if not self.significant_features_:
            raise ValueError("The model has not been fitted yet. Call `fit` first.")
        if drop_intercept and "Intercept" in self.significant_features_:
            return [f for f in self.significant_features_ if f != "Intercept"]
        return self.significant_features_

    def get_summary(self) -> pd.DataFrame:
        """
        Get the OLS model summary as a DataFrame.

        :return: A DataFrame containing coefficients, standard errors, t-values, and
            p-values.

        """
        if self.summary_ is None:
            raise ValueError("The model has not been fitted yet. Call `fit` first.")
        return self.summary_

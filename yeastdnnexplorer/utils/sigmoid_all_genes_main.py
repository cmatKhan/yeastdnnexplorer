import argparse
import json
import logging
import os

# import random
import re
import time
from typing import Any, Literal

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import run_app

from yeastdnnexplorer.ml_models.lasso_modeling import (
    bootstrap_stratified_cv_modeling,
    examine_bootstrap_coefficients,
    generate_modeling_data,
    stratification_classification,
    stratified_cv_modeling,
)
from yeastdnnexplorer.ml_models.sigmoid_modeling import GeneralizedLogisticModel
from yeastdnnexplorer.utils import LogLevel, configure_logger

logger = logging.getLogger("main")

# set seeds for reproduciblity
# random.seed(42)
# np.random.seed(42)


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

    # this is the first part of the code that checks for the edge case in which the main
    # effect isn't in the formula if it is not there, then we must add it when
    # generating the modeling data, and then ignore it for the actual modeling which
    # is done by the other part of the code in this method and in bootstrap_cv_modeling
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
        drop_intercept=False,
        formula=formula,  # Pass formula from kwargs
    )

    # NOTE: fit_intercept is set to `true`
    sigmoid_estimator = GeneralizedLogisticModel()

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
    sigmoid_model = stratified_cv_modeling(
        y, X, stratification_classes, sigmoid_estimator
    )

    if method == "lassocv_ols":
        # return a list of the non-zero features that survived the fitting
        non_zero_indices = sigmoid_model.coef_ != 0
        non_zero_features = X.columns[non_zero_indices]
        sig_coef_dict = {
            k: v
            for k, v in zip(non_zero_features, sigmoid_model.coef_[non_zero_indices])
        }

    elif method == "bootstrap_lassocv":

        # set the alphas_ attribute of the sigmoid_estimator to the alphas_
        # attribute of the sigmoid_model fit on the whole data. This will allow the
        # bootstrap_stratified_cv_modeling function to use the same set of lambdas
        sigmoid_estimator.alphas_ = sigmoid_model.alphas_

        # this is the third part specifically checks an edge case in sequential modeling
        # where after the lasso model on all genes is run and the main effect isn't
        # considered significant. Thus, we cannot consider it further, but we still need
        # to use it in this method to stratify datapoints for bootstrapping. This check
        # will enure that during bootstrap_stratified_cv_modeling we ultimately don't
        # also pass in the main effect
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
            estimator=sigmoid_estimator,
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


def configure_logging(
    log_level: int, handler_type: Literal["console", "file"] = "console"
) -> tuple[logging.Logger, logging.Logger]:
    """
    Configure the logging for the application.

    :param log_level: The logging level to set.
    :return: A tuple of the main and shiny loggers.

    """
    # add a timestamp to the log file name
    log_file = f"yeastdnnexplorer_{time.strftime('%Y%m%d-%H%M%S')}.log"
    main_logger = configure_logger(
        "main", level=log_level, handler_type=handler_type, log_file=log_file
    )
    shiny_logger = configure_logger(
        "shiny", level=log_level, handler_type=handler_type, log_file=log_file
    )
    return main_logger, shiny_logger


def run_shiny(args: argparse.Namespace) -> None:
    """
    Run the shiny app with the specified arguments.

    :param args: The parsed command-line arguments.

    """
    kwargs = {}
    if args.debug:
        kwargs["reload"] = True
        kwargs["reload_dirs"] = ["yeastdnnexplorer/shiny_app"]  # type: ignore
    app_import_string = "yeastdnnexplorer.shiny_app.app:app"
    run_app(app_import_string, **kwargs)


# this goes along with an example in the arg parser below, showing how to
# add cmd line utilies
# def run_another_command(args: argparse.Namespace) -> None:
#     """
#     Run another command with the specified arguments.

#     :param args: The parsed command-line arguments.
#     """
#     print(f"Running another command with parameter: {args.param}")


def run_lasso_bootstrap(args: argparse.Namespace) -> None:
    """
    Run GeneralizedLogisticModel() with bootstrap resampling on a specified
    transcription factor.

    :param args: The parsed command-line arguments.

    """
    output_dirpath = os.path.join(args.output_dir, args.response_tf)
    if os.path.exists(output_dirpath):
        raise FileExistsError(
            f"File {output_dirpath} already exists. "
            "Please specify a different `output_dir`."
        )
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        # Ensure the entire output directory path exists
        os.makedirs(output_dirpath, exist_ok=True)
    if not os.path.exists(args.response_file):
        raise FileNotFoundError(f"File {args.response_file} does not exist.")
    if not os.path.exists(args.predictors_file):
        raise FileNotFoundError(f"File {args.predictors_file} does not exist.")

    # Load data
    Y_filtered_transformed = pd.read_csv(args.response_file, index_col=0)
    predictors_df = pd.read_csv(args.predictors_file, index_col=0)

    # Generate modeling data
    y, X = generate_modeling_data(
        colname=args.response_tf,
        response_df=Y_filtered_transformed,
        predictors_df=predictors_df,
        drop_intercept=True,
        quantile_threshold=args.data_quantile,
        formula=args.formula,
    )

    # Configure and fit LassoCV estimator
    sigmoid_estimator = GeneralizedLogisticModel()

    if re.match("_rep\\d+", y.columns[0]):
        logger.debug(
            "Removing replicate suffix from the column name "
            "to create the stratification classes."
        )

    regulator_tf = re.sub("_rep\\d+", "", y.columns[0])

    try:
        classes = stratification_classification(X[regulator_tf].squeeze(), y.squeeze())
    except KeyError as exc:
        raise RuntimeError(
            f"column {regulator_tf} not found in predictors dataframe."
        ) from exc

    # Fit the model to extract alphas
    try:
        sigmoid_model = stratified_cv_modeling(y, X, classes, sigmoid_estimator)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fit the LassoCV model on {args.response_tf}."
        ) from exc

    # Set the alphas of the main estimator for bootstrap consistency
    sigmoid_estimator.alphas_ = sigmoid_model.alphas_

    # Run bootstrap modeling
    try:
        bootstrap_output = bootstrap_stratified_cv_modeling(
            y=y,
            X=X,
            estimator=sigmoid_estimator,
            bootstrap_cv=True,
            n_bootstraps=args.n_bootstraps,
            ci_percentiles=[90.0, 95.0, 99.0, 100.0],
            max_iter=10000,
            fit_intercept=True,
            selection="random",
            random_state=42,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to run bootstrap modeling on {args.response_tf}."
        ) from exc

    # the sigmoid_model and bootstrap_output in output_dirpath. boostrap_output is a
    # is a tuple where the first value is a dictionary called "ci_dict",
    # the second is a data frame called "bootstrap_coef_df",
    # and the third is a list called "bootstrap_alphas".

    # Save the fitted Lasso model
    sigmoid_model_path = os.path.join(output_dirpath, "sigmoid_model.joblib")
    joblib.dump(sigmoid_model, sigmoid_model_path)

    # Save ci_dict as JSON
    ci_dict = bootstrap_output[0]
    ci_dict_path = os.path.join(output_dirpath, "ci_dict.json")
    with open(ci_dict_path, "w") as f:
        json.dump(ci_dict, f, indent=4)

    # Save bootstrap_coef_df as CSV
    bootstrap_coef_df = bootstrap_output[1]
    bootstrap_coef_df_path = os.path.join(output_dirpath, "bootstrap_coef_df.csv")
    bootstrap_coef_df.to_csv(bootstrap_coef_df_path, index=False)

    # Save bootstrap_alphas as CSV
    bootstrap_alphas = pd.DataFrame(bootstrap_output[2], columns=["alpha"])
    bootstrap_alphas_path = os.path.join(output_dirpath, "bootstrap_alphas.csv")
    bootstrap_alphas.to_csv(bootstrap_alphas_path, index=False)


# 12/1: new attempt


def find_interactors_workflow(args: argparse.Namespace) -> None:
    """
    Run the find_interactors_workflow with the specified arguments.

    :param args: The parsed command-line arguments.

    """
    output_dirpath = os.path.join(args.output_dir, args.response_tf)
    if os.path.exists(output_dirpath):
        raise FileExistsError(
            f"Directory {output_dirpath} already exists. "
            "Please specify a different `output_dir`."
        )
    else:
        os.makedirs(output_dirpath, exist_ok=True)
    if not os.path.exists(args.response_file):
        raise FileNotFoundError(f"File {args.response_file} does not exist.")
    if not os.path.exists(args.predictors_file):
        raise FileNotFoundError(f"File {args.predictors_file} does not exist.")

    # Load data
    response_df = pd.read_csv(args.response_file, index_col=0)
    predictors_df = pd.read_csv(args.predictors_file, index_col=0)

    # Run LassoCV on all genes
    lasso_res = get_significant_predictors(
        args.method,
        args.response_tf,
        response_df,
        predictors_df,
        ci_percentile=args.all_ci_percentile,
        n_bootstraps=args.n_bootstraps,
        add_max_lrb=True,
    )

    # Save the results from bootstrapping if applicable
    if args.method == "bootstrap_lassocv":
        # Save ci_dict
        ci_dict = lasso_res["bootstrap_lasso_output"][0]
        ci_dict_path = os.path.join(output_dirpath, "ci_dict.json")
        with open(ci_dict_path, "w") as f:
            json.dump(ci_dict, f, indent=4)

        # Save bootstrap_coef_df
        bootstrap_coef_df = pd.DataFrame(lasso_res["bootstrap_lasso_output"][1])
        bootstrap_coef_df_path = os.path.join(output_dirpath, "bootstrap_coef_df.csv")
        bootstrap_coef_df.to_csv(bootstrap_coef_df_path, index=False)


class CustomHelpFormatter(argparse.HelpFormatter):
    """Custom help formatter to format the subcommands and general options sections."""

    # CustomHelpFormatter code remains the same as you provided


def add_general_arguments_to_subparsers(subparsers, general_arguments):
    for subparser in subparsers.choices.values():
        for arg in general_arguments:
            subparser._add_action(arg)


def main() -> None:
    """Main entry point for the YeastDNNExplorer application."""
    parser = argparse.ArgumentParser(
        prog="yeastdnnexplorer",
        description="YeastDNNExplorer Main Entry Point",
        usage="yeastdnnexplorer --help",
        formatter_class=CustomHelpFormatter,
    )

    formatter = parser._get_formatter()

    # Shared parameter for logging level
    log_level_argument = parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    log_handler_argument = parser.add_argument(
        "--log-handler",
        type=str,
        default="console",
        choices=["console", "file"],
        help="Set the logging handler",
    )
    formatter.add_arguments([log_level_argument, log_handler_argument])

    # Define subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Shiny command
    shiny_parser = subparsers.add_parser(
        "shiny",
        help="Run the shiny app",
        description="Run the shiny app",
        formatter_class=CustomHelpFormatter,
    )
    shiny_parser.add_argument(
        "--debug", action="store_true", help="Run the app with reloading enabled"
    )
    shiny_parser.set_defaults(func=run_shiny)

    # An example of adding another command
    # another_parser = subparsers.add_parser(
    #     "another_command",
    #     help="Run another command",
    #     description="Run another command",
    #     formatter_class=CustomHelpFormatter,
    # )
    # another_parser.add_argument(
    #     "--param", type=str, required=True, help="A parameter for another command"
    # )
    # another_parser.set_defaults(func=run_another_command)

    # Lasso Bootstrap command
    lasso_parser = subparsers.add_parser(
        "lasso_bootstrap",
        help="Run LassoCV with bootstrap resampling",
        description="Run LassoCV with bootstrap resampling",
        formatter_class=CustomHelpFormatter,
    )

    # Input arguments
    input_group = lasso_parser.add_argument_group("Input")
    input_group.add_argument(
        "--response_file",
        type=str,
        required=True,
        help="Path to the response CSV file. NOTE: the index column must be "
        "present and the first column in the CSV. Additionally, the index values "
        "should be either symbols or locus tags, matching the index values in both "
        "response and predictors files. The perturbed gene will be removed from "
        "the model data only if column names in response data match the index "
        "format (e.g., symbol or locus tag).",
    )
    input_group.add_argument(
        "--predictors_file",
        type=str,
        required=True,
        help="Path to the predictors CSV file. NOTE: the index column must be "
        "present and the first column in the CSV. Additionally, the index values "
        "should be either symbols or locus tags, matching the index values in both "
        "response and predictors files. The perturbed gene will be removed from the "
        "model data only if column names in predictors data match the index "
        "format (e.g., symbol or locus tag).",
    )
    input_group.add_argument(
        "--perturbed_tf",
        type=str,
        help="A response variable column to use. The data indices should be in "
        "the same format (e.g., symbol or locus tag) so that the perturbed gene can "
        "be removed from the data prior to modeling.",
    )
    input_group.add_argument(
        "--formula",
        type=str,
        default=None,
        help="The formula to use for modeling. If omitted, a formula with all of "
        "the interactors will be used, eg "
        "perturbed_tf_lrr ~ perturbed_tf + perturbed_tf:other_tf1 + ...",
    )
    input_group.add_argument(
        "--data_quantile",
        type=float,
        default=None,
        help="The quantile threshold for filtering the data based on the "
        "perturbed binding data. For example, 0.1 would select the top 10 percent. "
        "If omitted, all data will be used.",
    )
    input_group.add_argument(
        "--n_bootstraps",
        type=int,
        default=1000,
        help="Number of bootstrap samples to generate.",
    )

    # Output arguments
    output_group = lasso_parser.add_argument_group("Output")
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="./lasso_bootstrap_output",
        help="Path to the output directory where results will be saved.",
    )

    lasso_parser.set_defaults(func=run_lasso_bootstrap)

    # Find Interactors Workflow command
    find_interactors_parser = subparsers.add_parser(
        "find_interactors_workflow",
        help="Run the find interactors workflow",
        description="Run the find interactors workflow",
        formatter_class=CustomHelpFormatter,
    )

    # Input arguments
    input_group = find_interactors_parser.add_argument_group("Input")
    input_group.add_argument(
        "--response_file",
        type=str,
        required=True,
        help="Path to the response CSV file. NOTE: the index column must be "
        "present and the first column in the CSV. Additionally, the index values "
        "should be either symbols or locus tags, matching the index values in both "
        "response and predictors files. The perturbed gene will be removed from "
        "the model data only if column names in response data match the index "
        "format (e.g., symbol or locus tag).",
    )
    input_group.add_argument(
        "--predictors_file",
        type=str,
        required=True,
        help="Path to the predictors CSV file. NOTE: the index column must be "
        "present and the first column in the CSV. Additionally, the index values "
        "should be either symbols or locus tags, matching the index values in both "
        "response and predictors files. The perturbed gene will be removed from the "
        "model data only if column names in predictors data match the index "
        "format (e.g., symbol or locus tag).",
    )
    input_group.add_argument(
        "--response_tf",
        type=str,
        required=True,
        help="The response TF to use for the modeling.",
    )
    input_group.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["bootstrap_lassocv", "lassocv_ols"],
        help="The method to use for modeling.",
    )
    input_group.add_argument(
        "--all_ci_percentile",
        type=float,
        default=99.8,
        help="The percentile to use for the all model CI. This will only be used "
        "if the method is `bootstrap_lassocv`.",
    )
    input_group.add_argument(
        "--top_ci_percentile",
        type=float,
        default=90.0,
        help="The percentile to use for the top model CI. This will only be used "
        "if the method is `bootstrap_lassocv`.",
    )
    input_group.add_argument(
        "--all_pval_threshold",
        type=float,
        default=0.001,
        help="The p-value threshold to use for the all model. This will only be used "
        "if the method is `lassocv_ols`.",
    )
    input_group.add_argument(
        "--top_pval_threshold",
        type=float,
        default=0.01,
        help="The p-value threshold to use for the top model. This will only be used "
        "if the method is `lassocv_ols`.",
    )
    input_group.add_argument(
        "--data_quantile",
        type=float,
        default=0.1,
        help="The quantile threshold to use for the `top` data. See the tutorial for "
        "more information.",
    )
    input_group.add_argument(
        "--n_bootstraps",
        type=int,
        default=1000,
        help="Number of bootstrap samples to generate.",
    )

    # output arguments
    output_group = find_interactors_parser.add_argument_group("Output")
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="./find_interactors_output",
        help="Path to the output directory where results will be saved.",
    )

    find_interactors_parser.set_defaults(func=find_interactors_workflow)

    # Add the general arguments to the subcommand parsers
    add_general_arguments_to_subparsers(subparsers, [log_level_argument])

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    try:
        log_level = LogLevel.from_string(args.log_level)
    except ValueError as e:
        print(e)
        parser.print_help()
        return

    main_logger, shiny_logger = configure_logging(log_level)

    # Run the appropriate command
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()

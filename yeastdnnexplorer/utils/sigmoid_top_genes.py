import argparse
import json
import logging
import os

# import random
import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from shiny import run_app

from yeastdnnexplorer.ml_models.lasso_modeling import (
    bootstrap_stratified_cv_modeling,
    generate_modeling_data,
    stratification_classification,
    stratified_cv_modeling,
    stratified_cv_r2,
)
from yeastdnnexplorer.ml_models.sigmoid_modeling import GeneralizedLogisticModel
from yeastdnnexplorer.utils import LogLevel, configure_logger

logger = logging.getLogger("main")

# set seeds for reproduciblity
# random.seed(42)
# np.random.seed(42)


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


def get_interactor_importance(
    y: pd.DataFrame,
    full_X: pd.DataFrame,
    stratification_classes: np.ndarray,
    intersect_coefficients: list[str],
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
        y,
        full_X.loc[:, list(intersect_coefficients)],
        stratification_classes,
        GeneralizedLogisticModel(),
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


def try_interactor_variants(
    intersect_coefficients: list[str], interactor: str, **kwargs: Any
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
            y,
            X.loc[:, variant_predictors],
            stratification_classes,
            GeneralizedLogisticModel(),
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


def get_significant_predictors(conf: float, bootstrap_df: pd.DataFrame) -> list[str]:
    """
    Generate a list of features with non-crossing confidence intervals.

    Parameters:
        conf (float): Confidence level for confidence intervals (e.g., 95.0 for 95%
        confidence).
        bootstrap_df (pd.DataFrame): DataFrame containing bootstrap coefficients.

    Returns:
        list[str]: List of significant features whose CIs don't cross 0.

    """
    # Calculate confidence intervals for each feature
    intervals = {
        colname: (
            np.percentile(bootstrap_df[colname], (100 - conf) / 2),
            np.percentile(bootstrap_df[colname], 100 - (100 - conf) / 2),
        )
        for colname in bootstrap_df.columns
    }

    # Find features with intervals that don't cross zero
    significant_features = [
        key
        for key, (start, end) in intervals.items()
        if ((start > 0 and end > 0) or (start < 0 and end < 0)) and key != "Intercept"
    ]

    return significant_features


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

    # get the key and value
    key = args.response_tf

    bootstrap_dir = args.bootstrap_dir

    bootstrap_df = pd.read_csv(f"{bootstrap_dir}/{key}.csv")

    significant_predictors_all = get_significant_predictors(
        args.all_ci_percentile, bootstrap_df
    )

    # Build the formula using these predictors
    response_variable = key

    # Ensure the main effect of the perturbed TF is included
    formula_terms = significant_predictors_all.copy()

    # Build the formula string
    formula = f"{response_variable} ~ {' + '.join(formula_terms)}"

    lhs, rhs = formula.split("~")
    rhs = rhs.strip()
    rhs_terms = [term.strip() for term in rhs.split("+")]

    ignore_main_effect = False

    if key not in rhs_terms:
        formula = f"{lhs.strip()} ~ {rhs} + {key}"
        ignore_main_effect = True

    y, X = generate_modeling_data(
        key,
        response_df,
        predictors_df,
        quantile_threshold=0.1,
        drop_intercept=False,
        formula=formula,
    )

    sigmoid_estimator = GeneralizedLogisticModel()

    stratification_classes = stratification_classification(
        X[key].squeeze(), y.squeeze()
    )

    max_lrb = X.drop(columns=key).max(axis=1)
    if "max_lrb" in formula:
        X["max_lrb"] = max_lrb

    sigmoid_model = stratified_cv_modeling(
        y, X, stratification_classes, sigmoid_estimator
    )

    sigmoid_estimator.alphas_ = sigmoid_model.alphas_

    bootstrap_lasso_output = bootstrap_stratified_cv_modeling(
        y=y,
        X=X,
        estimator=sigmoid_estimator,
        ci_percentile=args.top_ci_percentile,
        n_bootstraps=args.n_bootstraps,
        max_iter=10000,
        fit_intercept=True,
        selection="random",
        ignore_main_effect=ignore_main_effect,
    )

    # Save bootstrap_coef_df
    bootstrap_coef_df = bootstrap_lasso_output[1]
    bootstrap_coef_df_path = os.path.join(
        output_dirpath, "bootstrap_top_genes_coef_df.csv"
    )
    bootstrap_coef_df.to_csv(bootstrap_coef_df_path, index=False)

    significant_predictors_top = get_significant_predictors(
        args.top_ci_percentile, bootstrap_coef_df
    )

    # Get main effects
    main_effects_seq = []
    for term in significant_predictors_top:
        if ":" in term:
            main_effects_seq.append(term.split(":")[1])
        else:
            main_effects_seq.append(term)

    # Combine main effects with final features
    interactor_terms_and_main_effects_seq = (
        list(significant_predictors_top) + main_effects_seq
    )

    # Generate model matrix
    _, full_X_seq = generate_modeling_data(
        args.response_tf,
        response_df,
        predictors_df,
        formula=f"~ {' + '.join(interactor_terms_and_main_effects_seq)}",
        drop_intercept=False,
    )

    # Add max_lrb column
    full_X_seq["max_lrb"] = max_lrb

    # Use the response and classes from the "all" data (NOT sequential data)
    y, X = generate_modeling_data(
        key,
        response_df,
        predictors_df,
        quantile_threshold=None,
        drop_intercept=False,
    )
    stratification_classes = stratification_classification(
        X[key].squeeze(), y.squeeze()
    )
    response_seq = y
    classes_seq = stratification_classes

    # Step 3: Get interactor importance using "all" data
    (
        full_avg_rsquared_seq,
        interactor_results_seq,
    ) = get_interactor_importance(
        response_seq,
        full_X_seq,
        classes_seq,
        significant_predictors_top,
    )

    # Update final features based on interactor results we remove all interactors whose
    # main effects improved r^2
    for interactor_variant in interactor_results_seq:
        k = interactor_variant["interactor"]
        significant_predictors_top.remove(k)

    # check if there are still significant predictors left - proceed if so
    if significant_predictors_top:
        final_preds = list(significant_predictors_top)
        # need to add intercept for sigmoid modeling
        final_preds.insert(0, "Intercept")

        final_model_avg_r_squared_seq = stratified_cv_r2(
            response_seq,
            full_X_seq[final_preds],
            classes_seq,
            estimator=GeneralizedLogisticModel(),
        )

        # Prepare output dictionary
        output_dict_seq = {
            "response_tf": args.response_tf,
            "final_features": final_preds,
            "final_model_avg_r_squared": final_model_avg_r_squared_seq,
        }

        # Save the output
        output_path_seq = os.path.join(output_dirpath, "final_output_sequential.json")
        with open(output_path_seq, "w") as f:
            json.dump(output_dict_seq, f, indent=4)


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
        "--bootstrap_dir",
        type=str,
        required=True,
        help="Path to the directory that contains all of the bootstrap csv files.",
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

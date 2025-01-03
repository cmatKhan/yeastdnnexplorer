import argparse
import json
import logging
import os
import random
import re
import time
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from shiny import run_app
from sklearn.linear_model import LassoCV

from yeastdnnexplorer.ml_models.lasso_modeling import (
    OLSFeatureSelector,
    bootstrap_stratified_cv_modeling,
    generate_modeling_data,
    get_interactor_importance,
    get_significant_predictors,
    stratification_classification,
    stratified_cv_modeling,
    stratified_cv_r2,
)
from yeastdnnexplorer.utils import LogLevel, configure_logger

logger = logging.getLogger("main")

# set seeds for reproduciblity
random.seed(42)
np.random.seed(42)


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
    kwargs["port"] = 8006  # type: ignore
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
    Run LassoCV with bootstrap resampling on a specified transcription factor.

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
    lassoCV_estimator = LassoCV(
        fit_intercept=True,
        max_iter=10000,
        selection="random",
        random_state=42,
        n_jobs=4,
    )

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
        lasso_model = stratified_cv_modeling(y, X, classes, lassoCV_estimator)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fit the LassoCV model on {args.response_tf}."
        ) from exc

    # Set the alphas of the main estimator for bootstrap consistency
    lassoCV_estimator.alphas_ = lasso_model.alphas_

    # Run bootstrap modeling
    try:
        bootstrap_output = bootstrap_stratified_cv_modeling(
            y=y,
            X=X,
            estimator=lassoCV_estimator,
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

    # the lasso_model and bootstrap_output in output_dirpath. boostrap_output is a
    # is a tuple where the first value is a dictionary called "ci_dict",
    # the second is a data frame called "bootstrap_coef_df",
    # and the third is a list called "bootstrap_alphas".

    # Save the fitted Lasso model
    lasso_model_path = os.path.join(output_dirpath, "lasso_model.joblib")
    joblib.dump(lasso_model, lasso_model_path)

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

    # Step 1: Run LassoCV, possibly with the bootstrap depending on args.method
    lasso_res = {
        "all": get_significant_predictors(
            args.method,
            args.response_tf,
            response_df,
            predictors_df,
            ci_percentile=args.all_ci_percentile,
            n_bootstraps=args.n_bootstraps,
            add_max_lrb=True,
        ),
        "top": get_significant_predictors(
            args.method,
            args.response_tf,
            response_df,
            predictors_df,
            ci_percentile=args.top_ci_percentile,
            n_bootstraps=args.n_bootstraps,
            add_max_lrb=True,
            quantile_threshold=args.data_quantile,
        ),
    }

    # Save the results from bootstrapping for further analysis
    # also perform the sequential processing where we take the
    # results from "all" and call get_significant_predictors
    # using only the TFs from these results on the top x% of data
    # Sequential analysis
    if args.method == "bootstrap_lassocv":
        # Iterate through "all" and "top"
        for suffix, lasso_results in lasso_res.items():
            # Save ci_dict
            ci_dict = lasso_results["bootstrap_lasso_output"][0]
            ci_dict_path = os.path.join(output_dirpath, f"ci_dict_{suffix}.json")
            with open(ci_dict_path, "w") as f:
                json.dump(ci_dict, f, indent=4)

            # Save bootstrap_coef_df
            bootstrap_coef_df = pd.DataFrame(lasso_results["bootstrap_lasso_output"][1])
            bootstrap_coef_df_path = os.path.join(
                output_dirpath, f"bootstrap_coef_df_{suffix}.csv"
            )
            bootstrap_coef_df.to_csv(bootstrap_coef_df_path, index=False)
        # Get significant predictors from "all" results
        assert isinstance(lasso_res["all"]["sig_coefs"], dict)
        significant_predictors_all = list(lasso_res["all"]["sig_coefs"].keys())

        # Build the formula using these predictors
        response_variable = f"{args.response_tf}_LRR"
        predictor_variable = re.sub(r"_rep\\d+", "", args.response_tf)

        # Ensure the main effect of the perturbed TF is included
        formula_terms = significant_predictors_all.copy()
        if predictor_variable not in formula_terms:
            formula_terms.insert(0, predictor_variable)

        # Build the formula string
        formula = f"{response_variable} ~ {' + '.join(formula_terms)}"

        # Add '-1' to the formula if drop_intercept is True
        formula += " - 1"

        # Run get_significant_predictors with the custom formula
        sequential_lasso_res = get_significant_predictors(
            args.method,
            args.response_tf,
            response_df,
            predictors_df,
            ci_percentile=args.top_ci_percentile,
            n_bootstraps=args.n_bootstraps,
            add_max_lrb=True,
            quantile_threshold=args.data_quantile,
            formula=formula,  # Pass the formula via kwargs
        )

        # Rest of the code remains the same...
        # Store the results under a new directory
        sequential_output_dir = os.path.join(
            output_dirpath, "sequential_top_genes_results"
        )
        os.makedirs(sequential_output_dir, exist_ok=True)

        # Save ci_dict and bootstrap_coef_df from sequential_lasso_res
        ci_dict_seq = sequential_lasso_res["bootstrap_lasso_output"][0]
        ci_dict_seq_path = os.path.join(
            sequential_output_dir, "ci_dict_sequential.json"
        )
        with open(ci_dict_seq_path, "w") as f:
            json.dump(ci_dict_seq, f, indent=4)

        bootstrap_coef_df_seq = pd.DataFrame(
            sequential_lasso_res["bootstrap_lasso_output"][1]
        )
        bootstrap_coef_df_seq_path = os.path.join(
            sequential_output_dir, "bootstrap_coef_df_sequential.csv"
        )
        bootstrap_coef_df_seq.to_csv(bootstrap_coef_df_seq_path, index=False)

    # Ensure lasso_res["all"]["sig_coefs"] and
    # lasso_res["top"]["sig_coefs"] are dictionaries
    all_sig_coefs = lasso_res["all"]["sig_coefs"]
    top_sig_coefs = lasso_res["top"]["sig_coefs"]

    # intersect with type checking
    if isinstance(all_sig_coefs, dict) and isinstance(top_sig_coefs, dict):
        lasso_intersect_coefs = set(all_sig_coefs.keys()).intersection(
            set(top_sig_coefs.keys())
        )
    else:
        raise TypeError(
            "Expected 'sig_coefs' to be dictionaries in 'all' and 'top', "
            f"but got {type(all_sig_coefs)} and {type(top_sig_coefs)}."
        )

    # extract the predictors and ensure that they are dataframes
    all_predictors = lasso_res["all"]["predictors"]
    top_predictors = lasso_res["top"]["predictors"]
    top_response = lasso_res["top"]["response"]

    if not isinstance(all_predictors, pd.DataFrame) or not isinstance(
        top_predictors, pd.DataFrame
    ):
        raise TypeError(
            "Expected 'predictors' to be dataframes in 'all' and 'top', "
            f"but got {type(all_predictors)} and {type(top_predictors)}."
        )
    if not isinstance(top_response, pd.DataFrame):
        raise TypeError(
            "Expected 'response' to be a dataframe in 'top', "
            f"but got {type(top_response)}."
        )

    # Step 2: find the intersect coefficients between the all and top models. This is
    # performed differently depending on args.method (see the tutorial)
    if args.method == "lassocv_ols":

        # Initialize the selector
        selector_all = OLSFeatureSelector(p_value_threshold=args.all_pval_threshold)

        # Transform the data to select only significant features
        selector_all.refine_features(
            all_predictors[list(lasso_intersect_coefs)],
            lasso_res["all"]["response"],
        )

        selector_top10 = OLSFeatureSelector(p_value_threshold=args.top_pval_threshold)

        _ = selector_top10.refine_features(
            top_predictors.loc[top_response.index, list(lasso_intersect_coefs)],
            top_response,
        )

        final_features = set(
            selector_all.get_significant_features(drop_intercept=True)
        ).intersection(selector_top10.get_significant_features(drop_intercept=True))

    else:
        final_features = lasso_intersect_coefs

    # Save the intersection coefficients as a dictionary

    intersection_path = os.path.join(output_dirpath, "intersection.json")
    with open(intersection_path, "w") as f:
        json.dump(list(lasso_intersect_coefs), f, indent=4)

    # Step 3: determine if the interactor predictor is significant compared to its
    # main effect
    # get the additional main effects which will be tested from the final_features
    main_effects = []
    for term in final_features:
        if ":" in term:
            main_effects.append(term.split(":")[1])
        else:
            main_effects.append(term)

    # combine these main effects with the final_features
    interactor_terms_and_main_effects = list(final_features) + main_effects

    # generate a model matrix with the intersect terms and the main effects. This full
    # model will not be used for modeling -- subsets of the columns will be, however.
    _, full_X = generate_modeling_data(
        args.response_tf,
        response_df,
        predictors_df,
        formula=f"~ {' + '.join(interactor_terms_and_main_effects)}",
        drop_intercept=False,
    )

    # Add the max_lrb column, just in case it is present in the final_predictors.
    # In this case, it is not.
    model_tf = re.sub("_rep\\d+", "", args.response_tf)
    max_lrb = predictors_df.drop(columns=model_tf).max(axis=1)
    full_X["max_lrb"] = max_lrb

    # Currently, this function tests each interactor term in the final_features
    # with two variants by replacing the interaction term with the main effect only, and
    # with the main effect + interactor. If either of the variants has a higher avg
    # r-squared than the intersect_model, then that variant is returned. In this case,
    # the original final_features are the best model.
    full_avg_rsquared, interactor_results = get_interactor_importance(
        lasso_res["all"]["response"],
        full_X,
        lasso_res["all"]["classes"],
        final_features,
    )

    # use the interactor_results to update the final_features
    for interactor_variant in interactor_results:
        k = interactor_variant["interactor"]
        v = interactor_variant["variant"]
        final_features.remove(k)
        final_features.add(v)

    # Step 4: compare the results of the final model with a univariate model
    avg_r2_univariate = stratified_cv_r2(
        lasso_res["all"]["response"],
        all_predictors[[model_tf]],
        lasso_res["all"]["classes"],
    )

    final_model_avg_r_squared = stratified_cv_r2(
        lasso_res["all"]["response"],
        full_X[list(final_features)],
        lasso_res["all"]["classes"],
    )

    output_dict = {
        "response_tf": args.response_tf,
        "final_features": list(final_features),
        "avg_r2_univariate": avg_r2_univariate,
        "final_model_avg_r_squared": final_model_avg_r_squared,
    }

    output_path = os.path.join(output_dirpath, "final_output.json")
    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=4)

    # If bootstrap_lassocv is the chosen method, need to repeat steps 3 and 4 for the
    # sequential results
    if args.method == "bootstrap_lassocv":
        # Step 3 and 4 for the sequential method

        # Get the significant coefficients from the sequential results
        sequential_sig_coefs = sequential_lasso_res["sig_coefs"]
        assert isinstance(sequential_sig_coefs, dict)
        sequential_final_features = set(sequential_sig_coefs.keys())

        # Save the sequential significant coefficients as a dictionary
        intersection_seq_path = os.path.join(
            sequential_output_dir, "intersection_sequential.json"
        )
        with open(intersection_seq_path, "w") as f:
            json.dump(list(sequential_final_features), f, indent=4)

        # Get main effects
        main_effects_seq = []
        for term in sequential_final_features:
            if ":" in term:
                main_effects_seq.append(term.split(":")[1])
            else:
                main_effects_seq.append(term)

        # Combine main effects with final features
        interactor_terms_and_main_effects_seq = (
            list(sequential_final_features) + main_effects_seq
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
        response_seq = lasso_res["all"]["response"]
        classes_seq = lasso_res["all"]["classes"]

        # Step 3: Get interactor importance using "all" data
        (
            full_avg_rsquared_seq,
            interactor_results_seq,
        ) = get_interactor_importance(
            response_seq,
            full_X_seq,
            classes_seq,
            sequential_final_features,
        )

        # Update final features based on interactor results (no change needed here)
        for interactor_variant in interactor_results_seq:
            k = interactor_variant["interactor"]
            v = interactor_variant["variant"]
            sequential_final_features.remove(k)
            sequential_final_features.add(v)

        # Step 4: Compare results of final model vs. a univariate model on "all" data
        assert isinstance(lasso_res["all"]["predictors"], pd.DataFrame)
        avg_r2_univariate_seq = stratified_cv_r2(
            response_seq,
            lasso_res["all"]["predictors"][[model_tf]],
            classes_seq,
        )

        final_model_avg_r_squared_seq = stratified_cv_r2(
            response_seq,
            full_X_seq[list(sequential_final_features)],
            classes_seq,
        )

        # Prepare output dictionary
        output_dict_seq = {
            "response_tf": args.response_tf,
            "final_features": list(sequential_final_features),
            "avg_r2_univariate": avg_r2_univariate_seq,
            "final_model_avg_r_squared": final_model_avg_r_squared_seq,
        }

        # Save the output
        output_path_seq = os.path.join(
            sequential_output_dir, "final_output_sequential.json"
        )
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

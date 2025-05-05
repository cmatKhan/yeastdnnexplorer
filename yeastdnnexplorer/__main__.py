import argparse
import fcntl
import json
import logging
import os
import random
import shutil
import time
from typing import Literal

import numpy as np

# import pandas as pd
# from filelock import FileLock, Timeout
from shiny import run_app
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold

from yeastdnnexplorer.ml_models.lasso_modeling import (
    BootstrappedModelingInputData,
    ModelingInputData,
    bootstrap_stratified_cv_modeling,
    evaluate_interactor_significance,
    stratification_classification,
)
from yeastdnnexplorer.ml_models.SigmoidModel import SigmoidModel
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
    run_app(app_import_string, **kwargs)


# this goes along with an example in the arg parser below, showing how to
# add cmd line utilies
# def run_another_command(args: argparse.Namespace) -> None:
#     """
#     Run another command with the specified arguments.

#     :param args: The parsed command-line arguments.
#     """
#     print(f"Running another command with parameter: {args.param}")


def perturbation_binding_modeling(args):
    """
    :param args: Command-line arguments containing input file paths and parameters.
    """
    if not isinstance(args.max_iter, int) or args.max_iter < 1:
        raise ValueError("The `max_iter` parameter must be a positive integer.")

    max_iter = int(args.max_iter)

    logger.info(f"estimator max_iter: {max_iter}.")

    logger.info("Step 1: Preprocessing")

    # validate input files/dirs
    if not os.path.exists(args.response_file):
        raise FileNotFoundError(f"File {args.response_file} does not exist.")
    if not os.path.exists(args.predictors_file):
        raise FileNotFoundError(f"File {args.predictors_file} does not exist.")
    if os.path.exists(args.output_dir):
        logger.warning(f"Output directory {args.output_dir} already exists.")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory created at {args.output_dir}")

    # the output subdir is where the output of this modeling run will be saved
    output_subdir = os.path.join(
        args.output_dir, os.path.join(args.perturbed_tf + args.output_suffix)
    )
    if os.path.exists(output_subdir):
        raise FileExistsError(
            f"Directory {output_subdir} already exists. "
            "Please specify a different `output_dir`."
        )
    else:
        os.makedirs(output_subdir, exist_ok=True)
        logger.info(f"Output subdirectory created at {output_subdir}")

    try:
        all_data_bootstrap_indicies = (
            BootstrappedModelingInputData.load_indices(args.all_data_bootstrap_indicies)
            if args.all_data_bootstrap_indicies
            else None
        )
    except FileNotFoundError:
        logger.error(
            f"Bootstrap indices file {args.all_data_bootstrap_indicies} not found."
        )
        raise

    try:
        topn_data_bootstrap_indicies = (
            BootstrappedModelingInputData.load_indices(
                args.topn_data_bootstrap_indicies
            )
            if args.topn_data_bootstrap_indicies
            else None
        )
    except FileNotFoundError:
        logger.error(
            f"Bootstrap indices file {args.topn_data_bootstrap_indicies} not found."
        )
        raise

    # if the bootstrap indicies are not provided, then set the number of bootstraps
    # to the value passed in via the command line
    all_data_n_bootstraps = None if all_data_bootstrap_indicies else args.n_bootstraps
    topn_data_n_bootstraps = None if topn_data_bootstrap_indicies else args.n_bootstraps

    # instantiate a estimator
    # NOTE: fit_intercept is set to `true`. This means the intercept WILL BE fit
    # DO NOT add a constant vector to the predictors.
    estimator = LassoCV(
        fit_intercept=True,
        selection="random",
        n_alphas=100,
        random_state=42,
        n_jobs=args.n_cpus,
        max_iter=max_iter,
    )

    input_data = ModelingInputData.from_files(
        response_path=args.response_file,
        predictors_path=args.predictors_file,
        perturbed_tf=args.perturbed_tf,
        feature_blacklist_path=args.blacklist_file,
        top_n=args.top_n,
    )

    logger.info("Step 2: Bootstrap LassoCV on all data, full interactor model")

    # Unset the top n masking -- we want to use all the data for the first round
    # modeling
    input_data.top_n_masked = False

    # extract a list of predictor variables, which are the columns of the predictors_df
    predictor_variables = input_data.predictors_df.columns.drop(input_data.perturbed_tf)

    # drop any variables which are in args.exclude_interactor_variables
    predictor_variables = [
        var
        for var in predictor_variables
        if var not in args.exclude_interactor_variables
    ]

    # create a list of interactor terms with the perturbed_tf as the first term
    interaction_terms = [
        f"{input_data.perturbed_tf}:{var}" for var in predictor_variables
    ]
    # Construct the full interaction formula, ie perturbed_tf + perturbed_tf:other_tf1 +
    # perturbed_tf:other_tf2 + ... .
    all_data_formula = f"{input_data.perturbed_tf} + {' + '.join(interaction_terms)}"

    if args.squared_pTF:
        # if --squared_pTF is passed, then add the squared perturbed TF to the formula
        squared_term = f"I({input_data.perturbed_tf} ** 2)"
        logger.info(f"Adding squared term to model formula: {squared_term}")
        all_data_formula += f" + {squared_term}"

    # if --row_max is passed, then add "row_max" to the formula
    if args.row_max:
        logger.info("Adding `row_max` to the all data model formula")
        all_data_formula += " + row_max"

    # if --add_model_variables is passed, then add the variables to the formula
    if args.add_model_variables:
        logger.info(
            f"Adding model variables to the all data model "
            f"formula: {args.add_model_variables}"
        )
        all_data_formula += " + " + " + ".join(args.add_model_variables)

    # log the formula
    logger.info(f"All data formula for the full interactor model: {all_data_formula}")

    # create the bootstrapped data.
    bootstrapped_data_all = BootstrappedModelingInputData(
        response_df=input_data.response_df,
        model_df=input_data.get_modeling_data(
            all_data_formula, add_row_max=args.row_max, drop_intercept=True
        ),
        n_bootstraps=all_data_n_bootstraps,
        bootstrap_indices=all_data_bootstrap_indicies,
    )

    if not all_data_bootstrap_indicies:
        all_data_indicies_output_file = os.path.join(
            output_subdir, "all_data_bootstrap_indices.json"
        )
        logger.info(
            "Saving the bootstrap indices for the all "
            f"data model to {all_data_indicies_output_file}"
        )
        bootstrapped_data_all.save_indices(all_data_indicies_output_file)
    else:
        logger.info(
            "Using the provided bootstrap indices for the all data model from "
            f"{args.all_data_bootstrap_indicies}"
        )

    logger.info(
        f"Running bootstrap LassoCV on all data with {args.n_bootstraps} bootstraps"
    )
    all_data_results = bootstrap_stratified_cv_modeling(
        bootstrapped_data_all,
        input_data.predictors_df[input_data.perturbed_tf],
        estimator=estimator,
        use_sample_weight_in_cv=args.use_weights_in_cv,
        ci_percentiles=[float(args.all_data_ci_level)],
        bin_by_binding_only=args.bin_by_binding_only,
        bins=args.bins,
    )

    # create the all data object output subdir
    all_data_output = os.path.join(output_subdir, "all_data_result_object")
    os.makedirs(all_data_output, exist_ok=True)

    logger.info(f"Serializing all data results to {all_data_output}")
    all_data_results.serialize("result_obj", all_data_output)

    # Extract the coefficients that are significant at the specified confidence level
    all_data_sig_coefs = all_data_results.extract_significant_coefficients(
        ci_level=args.all_data_ci_level,
    )

    logger.info(f"all_data_sig_coefs: {all_data_sig_coefs}")

    if not all_data_sig_coefs:
        logger.warning(
            f"No significant coefficients found at {args.all_data_ci_level}% "
            "confidence level. Exiting."
        )
        return

    # write all_data_sig_coefs to a json file
    all_data_ci_str = str(args.all_data_ci_level).replace(".", "-")
    all_data_output_file = os.path.join(
        output_subdir, f"all_data_significant_{all_data_ci_str}.json"
    )
    logger.info(f"Writing the all data significant results to {all_data_output_file}")
    with open(
        all_data_output_file,
        "w",
    ) as f:
        json.dump(all_data_sig_coefs, f, indent=4)

    logger.info(
        "Step 3: Running LassoCV on topn data with significant coefficients "
        "from the all data model"
    )

    # Create the formula for the topn modeling from the significant coefficients
    # NOTE: to remove the intercept, we need to add " -1 "
    topn_formula = f"{' + '.join(all_data_sig_coefs.keys())}"
    logger.info(f"Topn formula: {topn_formula}")

    # apply the top_n masking
    input_data.top_n_masked = True

    # Create the bootstrapped data for the topn modeling
    bootstrapped_data_top_n = BootstrappedModelingInputData(
        response_df=input_data.response_df,
        model_df=input_data.get_modeling_data(
            topn_formula, add_row_max=args.row_max, drop_intercept=True
        ),
        n_bootstraps=topn_data_n_bootstraps,
        bootstrap_indices=topn_data_bootstrap_indicies,
    )

    # If the bootstrap indicies are generated from the data, save them to a json file
    # so that they can be reused in subsequent runs
    if not topn_data_bootstrap_indicies:
        topn_indicies_output_file = os.path.join(
            output_subdir, "topn_bootstrap_indices.json"
        )
        logger.info(
            "Saving the bootstrap indices for the topn data "
            f"to {topn_indicies_output_file}"
        )
        bootstrapped_data_top_n.save_indices(topn_indicies_output_file)
    else:
        logger.info(
            "Using the provided bootstrap indices for the topn data from "
            f"{args.topn_data_bootstrap_indicies}"
        )

    logger.debug(
        f"Running bootstrap LassoCV on topn data with {args.n_bootstraps} bootstraps"
    )
    topn_results = bootstrap_stratified_cv_modeling(
        bootstrapped_data_top_n,
        input_data.predictors_df[input_data.perturbed_tf],
        estimator=estimator,
        use_sample_weight_in_cv=args.use_weights_in_cv,
        ci_percentiles=[float(args.topn_ci_level)],
    )

    # create the topn data object output subdir
    topn_output = os.path.join(output_subdir, "topn_result_object")
    os.makedirs(topn_output, exist_ok=True)

    logger.info(f"Serializing topn results to {topn_output}")
    topn_results.serialize("result_obj", topn_output)

    # extract the topn_results at the specified confidence level
    topn_output_res = topn_results.extract_significant_coefficients(
        ci_level=args.topn_ci_level
    )

    logger.info(f"topn_output_res: {topn_output_res}")

    if not topn_output_res:
        logger.warning(
            f"No significant coefficients found at {args.topn_ci_level}% "
            "confidence level. Exiting."
        )
        return

    # write topn_output_res to a json file
    topn_ci_str = str(args.topn_ci_level).replace(".", "-")
    topn_output_file = os.path.join(
        output_subdir, f"topn_significant_{topn_ci_str}.json"
    )
    logger.info(f"Writing the topn significant results to {topn_output_file}")
    with open(topn_output_file, "w") as f:
        json.dump(topn_output_res, f, indent=4)

    logger.info(
        "Step 4: Test the significance of the interactor terms that survive "
        "against the corresoponding main effect"
    )

    # unmask the data
    input_data.top_n_masked = False

    # calculate the statification classes for the perturbed TF (all data)
    alldata_classes = stratification_classification(
        input_data.predictors_df[input_data.perturbed_tf].squeeze(),
        input_data.response_df.squeeze(),
        bin_by_binding_only=args.bin_by_binding_only,
        bins=args.bins,
    )

    # test the significance of the interactor against the main effect
    results = evaluate_interactor_significance(
        input_data,
        stratification_classes=alldata_classes,
        model_variables=list(
            topn_results.extract_significant_coefficients(ci_level="90.0").keys()
        ),
    )

    output_significance_file = os.path.join(
        output_subdir, "interactor_vs_main_result.json"
    )
    logger.info(
        "Writing the final interactor significance "
        "results to {output_significance_file}"
    )
    results.serialize(output_significance_file)


# def create_sqlite_db(args: argparse.Namespace):
#     """
#     Create an empty SQLite database file with WAL mode and busy timeout settings.

#     :param db_path: Path to the SQLite database file to create.
#     :param overwrite: Whether to overwrite the existing file if it exists.

#     """
#     # validate that the containing directory exists. If not create it
#     if not os.path.exists(os.path.dirname(args.db_path)):
#         os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
#         logger.info(
#             f"Directory {os.path.dirname(args.db_path)} "
#             "created for SQLite database."
#         )

#     if os.path.exists(args.db_path):
#         if not args.overwrite:
#             logger.info(
#                 f"SQLite database already exists at {args.db_path}. "
#                 "Skipping creation."
#             )
#             return
#         else:
#             os.remove(args.db_path)
#             logger.info(f"Existing database at {args.db_path} removed.")

#     conn = sqlite3.connect(args.db_path, timeout=60, isolation_level=None)
#     conn.execute("PRAGMA journal_mode=WAL;")
#     conn.execute("PRAGMA busy_timeout = 60000;")
#     conn.close()
#     logger.info(
#         f"Created SQLite database at {args.db_path} with WAL mode and busy_timeout."
#     )


def create_database(
    args: argparse.Namespace,
    bootstrap_results_table_name: str = "bootstrap_results",
    mse_table_name: str = "mse_path",
):
    """
    Prepare a JSONL output directory and optionally clear an existing one.

    If `overwrite` is True, the existing directory at `args.db_path` is deleted.

    :param args: Argument namespace with 'db_path' and 'overwrite' attributes.
    :param bootstrap_results_table_name: File name for bootstrap results.
    :param mse_table_name: File name for MSE results.

    """
    if os.path.exists(args.db_path):
        if args.overwrite:
            shutil.rmtree(args.db_path)
            logger.info(f"Existing directory at {args.db_path} removed.")
        else:
            logger.info(
                f"Directory already exists at {args.db_path}. Skipping creation."
            )

    # Always recreate the directory if it was removed or didn't exist
    os.makedirs(args.db_path, exist_ok=True)
    logger.info(f"Directory {args.db_path} is ready for JSONL output.")

    # Touch the .jsonl files
    bootstrap_jsonl_path = os.path.join(
        args.db_path, f"{bootstrap_results_table_name}.jsonl"
    )
    mse_jsonl_path = os.path.join(args.db_path, f"{mse_table_name}.jsonl")

    for path in [bootstrap_jsonl_path, mse_jsonl_path]:
        open(path, "a").close()  # touch: create if doesn't exist
        logger.info(f"Initialized empty file: {path}")

    logger.info(
        f"JSONL files initialized:\n"
        f"- {bootstrap_jsonl_path}\n"
        f"- {mse_jsonl_path}"
    )


def insert_result(
    i: int,
    db_path: str,
    result_row: dict,
    max_retries: int = 50,
    retry_wait: int = 5,
):
    """
    Append a single result row to a JSONL (newline-delimited JSON) file using fcntl for
    concurrency control.

    :param i: Bootstrap index (used for jitter and logging).
    :param db_path: Path to the output JSONL file.
    :param result_row: Dictionary of results to write.
    :param max_retries: Max retries if file is locked or busy.
    :param retry_wait: Base wait time between retries.

    """
    rng = np.random.default_rng(seed=i)

    for attempt in range(max_retries):
        logger.debug(
            f"Attempting to write bootstrap {i} to {db_path} "
            f"(attempt {attempt + 1}/{max_retries})"
        )
        try:
            with open(db_path, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(json.dumps(result_row) + "\n")
                f.flush()
                os.fsync(f.fileno())  # ensure it's written to disk
                fcntl.flock(f, fcntl.LOCK_UN)

            logger.debug(f"Successfully wrote bootstrap {i} to {db_path}.")
            return
        except Exception as e:
            logger.warning(f"[{i}] Write failed with error: {e}. Retrying...")
            time.sleep(retry_wait + rng.uniform(0, 2))

    logger.error(
        f"Failed to write bootstrap {i} to {db_path} after {max_retries} retries."
    )


def sigmoid_bootstrap_worker(
    args: argparse.Namespace,
    bootstrap_results_table_name: str = "bootstrap_results",
    mse_table_name: str = "mse_path",
) -> None:

    # create input data similar to perturbed_binding_modeling(). There needs to be a
    # setting to decide whether to do "all data" or "top n" modeling

    # Select bootstrap index
    i = int(args.bootstrap_idx)

    # Load input data
    input_data = ModelingInputData.from_files(
        response_path=args.response_file,
        predictors_path=args.predictors_file,
        perturbed_tf=args.perturbed_tf,
        feature_blacklist_path=args.blacklist_file,
        top_n=args.top_n,
    )

    # Determine formula
    if input_data.top_n_masked:
        raise NotImplementedError("Top n masking is not implemented for sigmoid model.")
        # create formula from the significant coefficients calculated from the
        # database
        # results = BootstrapModelResults.from_db(
        #     args.db_path, bootstrap_results_table_name
        # )
        # topn_sig_coefs = results.extract_significant_coefficients(
        #     ci_level=args.ci_level
        # )
        # formula = " + ".join(topn_sig_coefs.keys())
    else:
        predictor_variables = input_data.predictors_df.columns.drop(args.perturbed_tf)
        predictor_variables = [
            var
            for var in predictor_variables
            if var not in args.exclude_interactor_variables
        ]
        interaction_terms = [
            f"{args.perturbed_tf}:{var}" for var in predictor_variables
        ]
        formula = f"{args.perturbed_tf} + {' + '.join(interaction_terms)}"

        if args.squared_pTF:
            formula += f" + I({args.perturbed_tf} ** 2)"
        if args.row_max:
            formula += " + row_max"
        if args.add_model_variables:
            formula += " + " + " + ".join(args.add_model_variables)

    logger.info(f"Model formula: {formula}")
    model_df = input_data.get_modeling_data(
        formula, add_row_max=args.row_max, drop_intercept=args.drop_intercept
    )

    bootstrap_indices = BootstrappedModelingInputData.load_indices(
        args.bootstrap_indices_file
    )
    bootstrap_data = BootstrappedModelingInputData(
        response_df=input_data.response_df,
        model_df=model_df,
        n_bootstraps=None,
        bootstrap_indices=bootstrap_indices,
        normalize_sample_weights=args.normalize_sample_weights,
    )

    _, _, sample_weights = bootstrap_data.get_bootstrap_sample(i)

    classes = stratification_classification(
        input_data.predictors_df[input_data.perturbed_tf].squeeze(),
        input_data.response_df.squeeze(),
        bin_by_binding_only=args.bin_by_binding_only,
        bins=args.bins,
    )

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)

    folds = list(skf.split(bootstrap_data.model_df, classes))

    estimator = SigmoidModel(warm_start=args.warm_start, alphas=args.alphas, cv=folds)
    logger.info(f"Fitting model for bootstrap {args.bootstrap_idx}")
    estimator.fit(
        bootstrap_data.model_df,
        bootstrap_data.response_df.values.ravel(),
        sample_weight=sample_weights,
        minimize_options=args.minimize_options,
    )

    result_row = {
        "bootstrap_idx": i,
        "alpha": estimator.alpha_,
        "final_training_score": estimator.score(
            bootstrap_data.model_df, bootstrap_data.response_df.values.ravel()
        ),
        "left_asymptote": estimator.left_asymptote_,
        "right_asymptote": estimator.right_asymptote_,
        **dict(zip(bootstrap_data.model_df.columns, estimator.coef_)),
    }

    insert_result(
        i,
        os.path.join(args.db_path, f"{bootstrap_results_table_name}.jsonl"),
        result_row,
    )

    # Save MSE path if present
    if hasattr(estimator, "mse_path_") and hasattr(estimator, "alphas_"):
        n_alphas, n_folds = estimator.mse_path_.shape
        for a_idx in range(n_alphas):
            for f_idx in range(n_folds):
                mse_row = {
                    "bootstrap_idx": i,
                    "alpha": estimator.alphas_[a_idx],
                    "fold": f_idx,
                    "mse": estimator.mse_path_[a_idx, f_idx],
                }
                insert_result(
                    i, os.path.join(args.db_path, f"{mse_table_name}.jsonl"), mse_row
                )

    logger.info(f"Completed bootstrap {i}")


class CustomHelpFormatter(argparse.HelpFormatter):
    """
    This could be used to customize the help message formatting for the argparse parser.

    Left as a placeholder.

    """


def parse_bins(s):
    try:
        return [np.inf if x == "np.inf" else int(x) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid bin value in '{s}'")


def parse_comma_separated_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_json_dict(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {e}")


# Allowed keys for method='L-BFGS-B' (excluding deprecated options)
LBFGSB_ALLOWED_KEYS = {
    "maxcor",  # int
    "ftol",  # float
    "gtol",  # float
    "eps",  # float or ndarray
    "maxfun",  # int
    "maxiter",  # int
    "maxls",  # int
    "finite_diff_rel_step",  # float or array-like or None
}


def parse_lbfgsb_options(s):
    try:
        opts = json.loads(s)
        if not isinstance(opts, dict):
            raise ValueError("Options must be a JSON object")

        unexpected_keys = set(opts) - LBFGSB_ALLOWED_KEYS
        if unexpected_keys:
            raise argparse.ArgumentTypeError(
                f"Unexpected keys in --minimize_options: {unexpected_keys}"
            )
        return opts
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {e}")
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


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
        "perturbation_binding_modeling",
        help="Run LassoCV or GeneralizedLogisticModel with bootstrap resampling",
        description=(
            "This executes the sequential workflow which models first  "
            "`perturbation ~ binding` on all of the data, then extracts the "
            "significant predictors and does the same thing on the `top n` data. "
            "Finally it evaluates the surviving interactor terms against the "
            "corresponding main effect."
        ),
        formatter_class=CustomHelpFormatter,
    )

    # Input arguments
    input_group = lasso_parser.add_argument_group("Input")
    # Input arguments
    input_group.add_argument(
        "--response_file",
        type=str,
        required=True,
        help=(
            "Path to the response CSV file. The first column must contain "
            "feature names or locus tags (e.g., gene symbols), matching the index "
            "format in both response and predictor files. The perturbed gene will "
            "be removed from the model data only if its column names match the "
            "index format."
        ),
    )

    input_group.add_argument(
        "--predictors_file",
        type=str,
        required=True,
        help=(
            "Path to the predictors CSV file. The first column must contain "
            "feature names or locus tags (e.g., gene symbols), ensuring consistency "
            "between response and predictor files. The perturbed gene will be "
            "removed from the model if predictor column names match the index format."
        ),
    )

    input_group.add_argument(
        "--perturbed_tf",
        type=str,
        required=True,
        help=(
            "Name of the perturbed transcription factor (TF) used as the "
            "response variable. It must match a column in the response file. The "
            "format should be consistent with the feature index (e.g., gene symbol "
            "or locus tag)."
        ),
    )

    input_group.add_argument(
        "--blacklist_file",
        type=str,
        default="",
        help=(
            "Optional file containing a list of features (one per line) to be excluded "
            "from the analysis. If omitted, no features will be blacklisted."
        ),
    )

    input_group.add_argument(
        "--all_data_bootstrap_indicies",
        type=str,
        default=None,
        help=(
            "Path to a JSON file containing the bootstrap indices for the all data "
            "model. If provided, these indices will be used instead of generating "
            "new ones. If not provided, new bootstrap indices will be generated "
            "and saved."
        ),
    )

    input_group.add_argument(
        "--topn_data_bootstrap_indicies",
        type=str,
        default=None,
        help=(
            "Path to a JSON file containing the bootstrap indices for the topn data "
            "model. If provided, these indices will be used instead of generating "
            "new ones. If not provided, new bootstrap indices will be generated "
            "and saved."
        ),
    )

    parameters_group = lasso_parser.add_argument_group("Parameters")

    parameters_group.add_argument(
        "--top_n",
        type=int,
        default=600,
        help=(
            "Number of features to retain in the second round of modeling. "
            "Default is 600"
        ),
    )

    parameters_group.add_argument(
        "--n_bootstraps",
        type=int,
        default=1000,
        help="Number of bootstrap samples to generate for resampling. Default is 1000",
    )

    parameters_group.add_argument(
        "--all_data_ci_level",
        type=float,
        default=98.0,
        help=(
            "Confidence interval threshold (in percent) for selecting significant "
            "coefficients. Default is 98.0"
        ),
    )

    parameters_group.add_argument(
        "--topn_ci_level",
        type=float,
        default=90.0,
        help=(
            "Confidence interval threshold for the second round of modeling. "
            "Default is 90.0"
        ),
    )

    parameters_group.add_argument(
        "--max_iter",
        type=int,
        default=10000,
        help=(
            "This controls the maximum number of iterations LassoCV may "
            "use in order to fit"
        ),
    )

    parameters_group.add_argument(
        "--use_weights_in_cv",
        action="store_true",
        help=(
            "Enable sample weighting in cross-validation based on bootstrap "
            "sample proportions."
        ),
    )

    parameters_group.add_argument(
        "--row_max",
        action="store_true",
        help=(
            "Include the row max as an additional predictor in the model matrix "
            "in the first round (all data) model."
        ),
    )

    parameters_group.add_argument(
        "--squared_pTF",
        action="store_true",
        help=(
            "Include the squared pTF as an additional predictor in the model matrix "
            "in the first round (all data) model."
        ),
    )

    parameters_group.add_argument(
        "--bin_by_binding_only",
        action="store_true",
        help=(
            "When creating stratification classes, use binding data only instead of "
            "both binding and perturbation data. The default is to use both."
        ),
    )

    parameters_group.add_argument(
        "--bins",
        type=parse_bins,
        default="0,8,64,512,np.inf",
        help=(
            "Comma-separated list of bin edges (integers or 'np.inf'). "
            "Default is --bins 0,8,12,np.inf"
        ),
    )

    parameters_group.add_argument(
        "--exclude_interactor_variables",
        type=parse_comma_separated_list,
        default=[],
        help=(
            "Comma-separated list of variables to exclude from the interactor terms. "
            "E.g. red_median,green_median"
        ),
    )

    parameters_group.add_argument(
        "--add_model_variables",
        type=parse_comma_separated_list,
        default=[],
        help=(
            "Comma-separated list of variables to add to the all_data model. "
            "E.g., red_median,green_median would be added as ... + red_median + "
            "green_median"
        ),
    )

    # Output arguments
    output_group = lasso_parser.add_argument_group("Output")

    output_group.add_argument(
        "--output_dir",
        type=str,
        default="./perturbation_binding_modeling_results",
        help=(
            "Directory where model results will be saved. A new subdirectory "
            "is created per run."
        ),
    )

    output_group.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help=(
            "The subdirectory will be named by the perturbed_tf. "
            "Use output_suffix to add a suffix to the subdirectory name."
        ),
    )

    system_group = lasso_parser.add_argument_group("System")

    system_group.add_argument(
        "--n_cpus",
        type=int,
        default=4,
        help=(
            "Number of CPUs to use for parallel processing each lassoCV call. "
            "Recommended 4"
        ),
    )

    lasso_parser.set_defaults(func=perturbation_binding_modeling)

    # Sigmoid worker cmds
    sigmoid_parser = subparsers.add_parser(
        "sigmoid_bootstrap_worker",
        help="Run a single bootstrap iteration of the sigmoid model",
        description=(
            "This executes a single bootstrap iteration of the sigmoid model."
        ),
        formatter_class=CustomHelpFormatter,
    )

    sigmoid_input_group = sigmoid_parser.add_argument_group("Input")

    sigmoid_input_group.add_argument(
        "--response_file",
        type=str,
        required=True,
        help=(
            "Path to the response CSV file. The first column must contain "
            "feature names or locus tags (e.g., gene symbols), matching the index "
            "format in both response and predictor files. The perturbed gene will "
            "be removed from the model data only if its column names match the "
            "index format."
        ),
    )

    sigmoid_input_group.add_argument(
        "--predictors_file",
        type=str,
        required=True,
        help=(
            "Path to the predictors CSV file. The first column must contain "
            "feature names or locus tags (e.g., gene symbols), ensuring consistency "
            "between response and predictor files. The perturbed gene will be "
            "removed from the model if predictor column names match the index format."
        ),
    )
    sigmoid_input_group.add_argument(
        "--perturbed_tf",
        type=str,
        required=True,
        help=(
            "Name of the perturbed transcription factor (TF) used as the "
            "response variable. It must match a column in the response file. The "
            "format should be consistent with the feature index (e.g., gene symbol "
            "or locus tag)."
        ),
    )
    sigmoid_input_group.add_argument(
        "--blacklist_file",
        type=str,
        default="",
        help=(
            "Optional file containing a list of features (one per line) to be excluded "
            "from the analysis. If omitted, no features will be blacklisted."
        ),
    )
    sigmoid_input_group.add_argument(
        "--bootstrap_indices_file",
        type=str,
        required=True,
        help=(
            "Path to a JSON file containing the bootstrap indices for the model. "
            "These indices will be used for resampling."
        ),
    )

    sigmoid_input_group.add_argument(
        "--bootstrap_idx",
        type=int,
        required=True,
        help=(
            "Bootstrap index to use for the current iteration. This should be "
            "an integer corresponding to the bootstrap sample."
        ),
    )

    sigmoid_parameters_group = sigmoid_parser.add_argument_group("Parameters")

    sigmoid_parameters_group.add_argument(
        "--top_n",
        type=int,
        default=None,
        help=(
            "This is the number of features to use for second round modeling. "
            "Defaults to `Non`, for the 'all_data' model. Set to eg 600 for top_n "
            "modeling"
        ),
    )
    sigmoid_parameters_group.add_argument(
        "--ci_level",
        type=float,
        default=98.0,
        help=(
            "Confidence interval threshold for the second round of modeling. "
            "Default is 98.0. Only applied if `--top_n` is set"
        ),
    )
    sigmoid_parameters_group.add_argument(
        "--drop_intercept",
        action="store_true",
        help=("Drop the intercept from the model. Default is False"),
    )
    sigmoid_parameters_group.add_argument(
        "--warm_start",
        action="store_true",
        help=("Enable warm start for the model. Default is False"),
    )
    sigmoid_parameters_group.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0],
        help=(
            "List of alpha values to use for the model. " "Default is [0.1, 1.0, 10.0]"
        ),
    )
    sigmoid_parameters_group.add_argument(
        "--bin_by_binding_only",
        action="store_true",
        help=(
            "When creating stratification classes, use binding data only instead of "
            "both binding and perturbation data. The default is to use both."
        ),
    )

    sigmoid_parameters_group.add_argument(
        "--bins",
        type=parse_bins,
        default="0,8,64,512,np.inf",
        help=(
            "Comma-separated list of bin edges (integers or 'np.inf'). "
            "Default is --bins 0,8,12,np.inf"
        ),
    )

    sigmoid_parameters_group.add_argument(
        "--row_max",
        action="store_true",
        help=(
            "Include the row max as an additional predictor in the model matrix "
            "in the first round (all data) model."
        ),
    )

    sigmoid_parameters_group.add_argument(
        "--squared_pTF",
        action="store_true",
        help=(
            "Include the squared pTF as an additional predictor in the model matrix "
            "in the first round (all data) model."
        ),
    )

    sigmoid_parameters_group.add_argument(
        "--exclude_interactor_variables",
        type=parse_comma_separated_list,
        default=[],
        help=(
            "Comma-separated list of variables to exclude from the interactor terms. "
            "E.g. red_median,green_median"
        ),
    )

    sigmoid_parameters_group.add_argument(
        "--add_model_variables",
        type=parse_comma_separated_list,
        default=[],
        help=(
            "Comma-separated list of variables to add to the all_data model. "
            "E.g., red_median,green_median would be added as ... + red_median + "
            "green_median"
        ),
    )
    sigmoid_parameters_group.add_argument(
        "--minimize_options",
        type=parse_lbfgsb_options,
        help=(
            "JSON string of options for scipy.optimize.minimize with "
            "method='L-BFGS-B'. Allowed keys (with defaults): "
            "maxcor=10, ftol=2.22e-9, gtol=1e-5, eps=1e-8, maxfun=15000, "
            "maxiter=15000, maxls=20, finite_diff_rel_step=None. "
            'Example: \'{"maxiter": 1000, "gtol": 1e-6}\''
        ),
    )

    sigmoid_parameters_group.add_argument(
        "--normalize_sample_weights",
        action="store_true",
        help=(
            "Set this to normalize the sample weights to sum to 1. " "Default is False."
        ),
    )

    sigmoid_output_group = sigmoid_parser.add_argument_group("Output")

    sigmoid_output_group.add_argument(
        "--db_path",
        type=str,
        required=True,
        help=("Path to the database file where the results will be stored."),
    )

    sigmoid_parser.set_defaults(func=sigmoid_bootstrap_worker)

    # add create_database command
    create_db_parser = subparsers.add_parser(
        "create_database",
        help="Create an database file (sqlite or csv)",
        description=(
            "Create an empty database file with WAL mode and busy timeout " "settings."
        ),
        formatter_class=CustomHelpFormatter,
    )

    create_db_parser.add_argument(
        "--db_path",
        type=str,
        required=True,
        help="Path to the database file to create.",
    )

    create_db_parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite the existing database file if it exists. " "Default is False."
        ),
    )

    create_db_parser.set_defaults(func=create_database)

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

# %%

import plotly.graph_objects as go
from callingcardstools.Analysis.yeast.rank_response import compute_rank_response
from scipy.stats import binom


# Function to calculate confidence interval for the random value
def binom_ci(trials, random_prob, alpha=0.05):
    lower_bound = binom.ppf(alpha / 2, trials, random_prob) / trials
    upper_bound = binom.ppf(1 - alpha / 2, trials, random_prob) / trials
    return lower_bound, upper_bound


def process_plot_data(key, data):
    """Process a single row of metadata to generate plot data."""
    subset_data = data[data["rank_bin"] <= 150]
    rr_summary = compute_rank_response(subset_data)

    plot_data = {
        "x": rr_summary["rank_bin"],
        "y": rr_summary["response_ratio"],
        "random_y": rr_summary["random"] if "random" in rr_summary else None,
        "ci": (
            rr_summary["rank_bin"].apply(lambda n: binom_ci(n, rr_summary["random"][0]))
            if "random" in rr_summary
            else None
        ),
    }

    return plot_data


def prepare_rank_response_data(rr_dict):
    """Prepare rank response data for plotting."""
    metadata = rr_dict.get("metadata")
    data_dict = rr_dict.get("data")

    # Use list comprehension to generate plots
    plots: dict = {}
    for _, row in metadata.iterrows():
        id = str(row["id"])
        data = data_dict.get(id)

        expression_id = str(row["expression"])
        promotersetsig_id = str(row["promotersetsig"])

        plots.setdefault(expression_id, {}).update(
            {promotersetsig_id: process_plot_data(promotersetsig_id, data)}
        )

    return plots


def add_traces_to_plot(fig, sample_id, add_random, **kwargs):
    """Add traces to a Plotly figure based on plot data."""
    # Add the main line for the promoterset signal
    fig.add_trace(
        go.Scatter(
            x=kwargs["x"],
            y=kwargs["y"],
            mode="lines",
            name=f"{sample_id}",
            legendrank=int(sample_id),
        )
    )

    if add_random:
        # Add the random line
        fig.add_trace(
            go.Scatter(
                x=kwargs["x"],
                y=kwargs["random_y"],
                mode="lines",
                name="Random",
                line=dict(dash="dash", color="black"),
                legendrank=0,
            )
        )

        if kwargs["ci"] is not None:
            ci_lower = kwargs["ci"].apply(lambda x: x[0])
            ci_upper = kwargs["ci"].apply(lambda x: x[1])

            # Add confidence interval lower bound
            fig.add_trace(
                go.Scatter(
                    x=kwargs["x"],
                    y=ci_lower,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )

            # Add confidence interval upper bound and shade the area
            fig.add_trace(
                go.Scatter(
                    x=kwargs["x"],
                    y=ci_upper,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(128, 128, 128, 0.3)",
                    line=dict(width=0),
                    showlegend=False,
                )
            )


def create_rank_response_replicate_plot(plots_dict):
    """Generate a dictionary of Plotly figures from the prepared rank response data."""
    output_dict = {}

    for expression_id, promotersetsig_dict in plots_dict.items():
        fig = go.Figure()
        add_random = True
        for promotersetsig_id, plot_data in promotersetsig_dict.items():
            # Use the helper function to add traces to the plot
            add_traces_to_plot(fig, promotersetsig_id, add_random, **plot_data)
            add_random = False  # Add random line only once

        # Update the layout of the figure
        fig.update_layout(
            title={
                "text": f"Rank Response for Expression ID {expression_id}",
                "x": 0.5,
            },
            yaxis_title="# Responsive / # Genes",
            xaxis_title="Number of Genes, Ranked by Binding Score",
            xaxis=dict(tick0=0, dtick=5, range=[0, 150]),  # Set x-axis ticks and range
            yaxis=dict(
                tick0=0, dtick=0.1, range=[0, 1.0]
            ),  # Set y-axis ticks and range
        )

        output_dict[expression_id] = fig

    return output_dict


# ## Example

# # %%
# # Set up the environment
# import dotenv

# from yeastdnnexplorer.interface import *

# dotenv.load_dotenv("/home/chase/code/yeastdnnexplorer/.env", override=True)

# # configure the logger to print to console
# import logging

# logging.basicConfig(level=logging.DEBUG)

# rr_api = RankResponseAPI()

# # %%
# rr_api.pop_params()
# rr_api.push_params(
#     {
#         "regulator_symbol": "OAF1",
#     }
# )

# rr_dict = await rr_api.read()

# # %%
# plots = prepare_rank_response_data(rr_dict)

# # %%
# x = create_rank_response_replicate_plot(plots)

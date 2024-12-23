# %%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_rank_response_overview_plot(df: pd.DataFrame) -> go.Figure:
    """
    Create an enhanced rank response overview plot with interactive features.

    Group by binding_source and expression_source and create a Plotly boxplot of the
    rank_25 values with the individual points overlaid. Users can toggle points overlay
    and select a point to view its metadata.

    :param df: the rank response metadata table

    :return: a Plotly figure
    """
    fig = go.Figure()

    color_palette = px.colors.qualitative.Plotly

    for i, ((binding_source, expression_source), group) in enumerate(
        df.groupby(["binding_source", "expression_source"])
    ):
        fig.add_trace(
            go.Box(
                x=group["binding_source"],
                y=group["rank_25"],
                name=f"{binding_source} | {expression_source}",
                legendgroup=binding_source,
                legendgrouptitle=dict(text=binding_source),
                boxpoints="outliers",
                pointpos=0,
                jitter=0.5,
                text=group["id"],
                marker=dict(color=color_palette[i % len(color_palette)]),
                line=dict(color=color_palette[i % len(color_palette)]),
            )
        )

    # Add toggle for point visibility
    fig.update_layout(
        title="Rank 25 Values by Binding Source and Expression Source",
        xaxis_title="Binding Source",
        yaxis_title="Rank 25",
        boxmode="group",
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                buttons=[
                    dict(
                        label="Show Points",
                        method="update",
                        args=[{"boxpoints": "all"}],
                    ),
                    dict(
                        label="Hide Points",
                        method="update",
                        args=[{"boxpoints": False}],
                    ),
                ],
            )
        ],
        hovermode="x unified",
        legend=dict(groupclick="toggleitem"),
        xaxis=dict(gridcolor="lightgray", zerolinecolor="gray"),
        yaxis=dict(gridcolor="lightgray", zerolinecolor="gray"),
    )

    return fig


## Example

# %%
# Set up the environment
# import dotenv

# from yeastdnnexplorer.interface import *

# dotenv.load_dotenv("/home/chase/code/yeastdnnexplorer/.env", override=True)

# # configure the logger to print to console
# import logging

# logging.basicConfig(level=logging.DEBUG)

# rr_api = RankResponseAPI()

# rr_dict = await rr_api.read()

# plot = create_rank_response_overview_plot(rr_dict.get("metadata"))

# %%

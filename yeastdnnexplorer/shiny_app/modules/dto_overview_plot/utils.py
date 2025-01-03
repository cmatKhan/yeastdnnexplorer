# %%
import json
import re
from typing import Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def parse_dto_metadata(dto_meta: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:

    if not inplace:
        dto_meta = dto_meta.copy()

    # Assuming dto_meta is a pandas DataFrame
    dto_meta["json_parsed"] = dto_meta["result"].apply(
        lambda x: json.loads(re.sub("'", '"', x))
    )

    dto_meta["fdr"] = dto_meta["json_parsed"].apply(lambda x: x.get("fdr", None))
    dto_meta["empirical_pvalue"] = dto_meta["json_parsed"].apply(
        lambda x: x.get("empirical_pvalue", None)
    )
    dto_meta["binding_set_len"] = dto_meta["json_parsed"].apply(
        lambda x: x.get("set1_len", None)
    )
    dto_meta["perturb_set_len"] = dto_meta["json_parsed"].apply(
        lambda x: x.get("set2_len", None)
    )
    dto_meta["intersection_size"] = dto_meta["json_parsed"].apply(
        lambda x: x.get("unpermuted_intersection_size", None)
    )

    return dto_meta


def create_dto_overview_plot(
    df: pd.DataFrame, measure_colname: Literal["empirical_pvalue", "fdr"]
) -> go.Figure:
    """
    Create an enhanced rank response overview plot with interactive features.

    Group by binding_source and expression_source and create a Plotly boxplot of the
    rank_25 values with the individual points overlaid. Users can toggle points overlay
    and select a point to view its metadata.

    :param df: the rank response metadata table
    :param measure_colname: the column name of the measure to plot
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
                y=group[measure_colname],
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
        title=measure_colname + " by Binding Source and Expression Source",
        xaxis_title="Binding Source",
        yaxis_title=measure_colname,
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


# Example

# %%
# Set up the environment
# import dotenv

# from yeastdnnexplorer.interface import *

# dotenv.load_dotenv("/home/chase/code/yeastdnnexplorer/.env", override=True)

# # configure the logger to print to console
# import logging

# logging.basicConfig(level=logging.DEBUG)

# dto_api = DtoAPI()

# dto_res = await dto_api.read()

# parse_dto_metadata(dto_res.get("metadata"), inplace=True)

# df = dto_res.get("metadata")

# plot = create_dto_overview_plot(df, "fdr")

# %%

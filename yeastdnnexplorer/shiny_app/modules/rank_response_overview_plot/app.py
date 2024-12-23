import logging

import numpy as np
import pandas as pd
from shiny import App, reactive, run_app, ui

from yeastdnnexplorer.shiny_app.modules.rank_response_overview_plot.module import (
    rank_response_overview_plot_server,
    rank_response_overview_plot_ui,
)
from yeastdnnexplorer.utils import configure_logger

logger = logging.getLogger("shiny")

configure_logger("shiny")

app_ui = ui.page_fluid(
    rank_response_overview_plot_ui("rank_response_overview_plot"),
)


def app_server(input, output, session):
    # Parameters for the mock dataset

    num_rows = 1000  # Approximate number of rows

    binding_sources = ["harbison_chip", "chipexo_pugh_allevents", "brent_nf_cc"]

    expression_sources = ["mcisaac_oe", "kemmeren_tfko", "hu_reimann_tfko"]

    # Randomly generate data

    np.random.seed(42)  # For reproducibility

    data = {
        "id": np.arange(1, num_rows + 1),
        "binding_source": np.random.choice(binding_sources, size=num_rows),
        "expression_source": np.random.choice(expression_sources, size=num_rows),
        "rank_25": np.round(np.random.uniform(0, 1, size=num_rows), 2),
    }

    # Create DataFrame

    _rr_meta = reactive.Value(pd.DataFrame(data))

    rank_response_overview_plot_server("rank_response_overview_plot", _rr_meta)


app = App(ui=app_ui, server=app_server)

if __name__ == "__main__":
    run_app(
        "yeastdnnexplorer.shiny_app.modules.rank_response_overview_plot.app:app",
        reload=True,
        reload_dirs=["."],
        port=8005,
    )

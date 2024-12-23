import logging

import numpy as np
import pandas as pd
from shiny import App, reactive, run_app, ui

from yeastdnnexplorer.shiny_app.modules.dto_overview_plot.module import (
    dto_overview_plot_server,
    dto_overview_plot_ui,
)
from yeastdnnexplorer.utils import configure_logger

logger = logging.getLogger("shiny")

configure_logger("shiny")

app_ui = ui.page_fluid(
    dto_overview_plot_ui("dto_overview_plot"),
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
        "empirical_pvalue": np.round(np.random.uniform(0, 1, size=num_rows), 2),
        "fdr": np.round(np.random.uniform(0, 7, size=num_rows), 2),
    }

    # Create DataFrame

    _dto_meta = reactive.Value(pd.DataFrame(data))

    dto_overview_plot_server("dto_overview_plot", _dto_meta)


app = App(ui=app_ui, server=app_server)

if __name__ == "__main__":
    run_app(
        "yeastdnnexplorer.shiny_app.modules.dto_overview_plot.app:app",
        reload=True,
        reload_dirs=["."],
        port=8005,
    )

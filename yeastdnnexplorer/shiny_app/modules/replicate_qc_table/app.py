import logging

import pandas as pd
from shiny import App, reactive, run_app, ui

from yeastdnnexplorer.shiny_app.modules.replicate_qc_table.module import (
    replicate_qc_table_server,
    replicate_qc_table_ui,
)
from yeastdnnexplorer.utils import configure_logger

logger = logging.getLogger("shiny")

configure_logger("shiny")

app_ui = ui.page_fluid(
    replicate_qc_table_ui("replicate_qc_table"),
)


def app_server(input, output, session):

    _rr_meta = reactive.Value(pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}))

    replicate_qc_table_server("replicate_qc_table", _rr_meta)


app = App(ui=app_ui, server=app_server)

if __name__ == "__main__":
    run_app(
        "yeastdnnexplorer.shiny_app.modules.replicate_qc_table.app:app",
        reload=True,
        reload_dirs=["."],
        port=8005,
    )

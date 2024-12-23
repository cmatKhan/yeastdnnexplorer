import logging

import pandas as pd
from shiny import Inputs, Outputs, Session, module, reactive, render, req, ui

logger = logging.getLogger("shiny")


@module.ui
def replicate_qc_table_ui():
    return ui.div(
        ui.output_data_frame("replicate_qc_table"),
    )


@module.server
def replicate_qc_table_server(
    input: Inputs, output: Outputs, session: Session, _rr_res: reactive.Value
):

    height = 350
    width = "fit-content"

    @reactive.effect
    def _():
        rr_res = _rr_res.get()
        df = rr_res.get("metadata", pd.DataFrame())
        logger.info("Rendering replicate QC table")

        @render.data_frame
        def replicate_qc_table():
            return render.DataTable(
                df,
                width=width,
                height=height,
                filters=True,
                editable=True,
                summary=True,
                selection_mode="rows",
            )

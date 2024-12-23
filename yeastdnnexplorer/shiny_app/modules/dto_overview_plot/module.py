import logging

from shiny import Inputs, Outputs, Session, module, reactive, render, req, ui
from shinywidgets import output_widget, render_plotly

from .utils import create_dto_overview_plot

logger = logging.getLogger("shiny")


@module.ui
def dto_overview_plot_ui():
    return ui.row(
        ui.column(
            6,
            ui.card(
                ui.card_header("P-Value Distribution"),
                ui.input_slider(
                    "pval_range", "P-Value Range:", min=0, max=1, value=0.05
                ),
                output_widget("pval_plot"),
                ui.card_footer("Adjust the range to filter the plot."),
                style="margin-bottom: 20px; padding: 15px; border: 1px solid #ddd;",
            ),
        ),
        ui.column(
            6,
            ui.card(
                ui.card_header("FDR Distribution"),
                ui.input_slider("fdr_range", "FDR Range:", min=0, max=1, value=0.1),
                output_widget("fdr_plot"),
                ui.card_footer("Use the slider to refine the plot view."),
                style="margin-bottom: 20px; padding: 15px; border: 1px solid #ddd;",
            ),
        ),
    )


@module.server
def dto_overview_plot_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    trigger: reactive.value,
    _dto_meta: reactive.calc,
):

    @reactive.effect
    @reactive.event(trigger)
    def _():
        dto_meta = _dto_meta()
        logger.info("Rendering rank response overview plot")
        pval_fig = create_dto_overview_plot(dto_meta, "empirical_pvalue")
        fdr_fig = create_dto_overview_plot(dto_meta, "fdr")

        @render_plotly
        def pval_plot():
            return pval_fig

        @render_plotly
        def fdr_plot():
            return fdr_fig

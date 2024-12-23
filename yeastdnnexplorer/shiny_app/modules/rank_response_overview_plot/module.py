import logging

from shiny import Inputs, Outputs, Session, module, reactive, render, req, ui
from shinywidgets import output_widget, render_plotly

from .utils import create_rank_response_overview_plot

logger = logging.getLogger("shiny")


@module.ui
def rank_response_overview_plot_ui():
    return ui.div(
        output_widget("rank_response_overview_plot"),
    )


@module.server
def rank_response_overview_plot_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    trigger: reactive.value,
    _rr_meta: reactive.calc,
):

    @reactive.effect
    @reactive.event(trigger)
    def _():
        rr_meta = _rr_meta()
        logger.info("Rendering rank response overview plot")
        fig = create_rank_response_overview_plot(rr_meta)

        @render_plotly
        def rank_response_overview_plot():
            return fig

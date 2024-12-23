import logging

from shiny import App, run_app, ui

from yeastdnnexplorer.shiny_app.modules.rank_response_replicate_plot.module import (
    rank_response_replicate_plot_server,
    rank_response_replicate_plot_ui,
)
from yeastdnnexplorer.utils import configure_logger

logger = logging.getLogger("shiny")

configure_logger("shiny")

app_ui = ui.page_fluid(
    rank_response_replicate_plot_ui("rank_response_replicate_plot"),
)


def app_server(input, output, session):
    rank_response_replicate_plot_server("rank_response_replicate_plot")


app = App(ui=app_ui, server=app_server)

if __name__ == "__main__":
    run_app(
        "yeastdnnexplorer.shiny_app.modules.rank_response_replicate_plot.app:app",
        reload=True,
        reload_dirs=["."],
        port=8005,
    )

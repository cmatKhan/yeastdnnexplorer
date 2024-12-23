import asyncio
import logging

from shiny import App, reactive, run_app, ui

from yeastdnnexplorer.interface import (
    DtoAPI,
    ExpressionAPI,
    GenomicFeatureAPI,
    PromoterSetSigAPI,
    RankResponseAPI,
    RegulatorAPI,
)
from yeastdnnexplorer.shiny_app.modules.dataset_filters.module import (
    dataset_filters_server,
    dataset_filters_ui,
)
from yeastdnnexplorer.shiny_app.modules.dto_overview_plot import (
    dto_overview_plot_server,
    dto_overview_plot_ui,
    parse_dto_metadata,
)
from yeastdnnexplorer.shiny_app.modules.rank_response_overview_plot.module import (
    rank_response_overview_plot_server,
    rank_response_overview_plot_ui,
)
from yeastdnnexplorer.shiny_app.modules.rank_response_replicate_plot.module import (
    rank_response_replicate_plot_server,
    rank_response_replicate_plot_ui,
)
from yeastdnnexplorer.shiny_app.modules.replicate_qc_table.module import (
    replicate_qc_table_server,
    replicate_qc_table_ui,
)
from yeastdnnexplorer.shiny_app.modules.upset_plot.module import (
    upset_plot_server,
    upset_plot_ui,
)
from yeastdnnexplorer.utils import configure_logger

logger = logging.getLogger("shiny")

# Call the logger configuration function
configure_logger("shiny", level=logging.DEBUG)

app_ui = ui.page_sidebar(
    dataset_filters_ui("data_filters"),
    upset_plot_ui("upset_plot"),
    rank_response_overview_plot_ui("rank_response_overview_plot"),
    dto_overview_plot_ui("dto_overview_plot"),
    rank_response_replicate_plot_ui("rank_response_replicate_plot"),
    replicate_qc_table_ui("replicate_qc_table"),
)


def app_server(input, output, session):

    # A dictionary of reactive values to store the metadata from the APIs
    metadata_reactives = {
        "dto": reactive.Value(),
        "expression": reactive.Value(),
        "genomicfeature": reactive.Value(),
        "promotersetsig": reactive.Value(),
        "rankresponse": reactive.Value(),
        "regulator": reactive.Value(),
    }

    # this is a task that will pull the metadata from the APIs when the "pull data"
    # (see the data filter ui) button is clicked. This follows the pattern on the
    # python shiny async tasks example
    # https://shiny.posit.co/py/docs/nonblocking.html
    # without more threads on both the client and server side, and lots of bandwidth,
    # this probably isn't any faster.
    # NOTE: need to be careful -- this is set up with caching right now at the /export
    # endpoints. Need to make sure filtering doesn't affect this (i expect it does 20241220)
    @ui.bind_task_button(button_id="btn")
    @reactive.extended_task
    async def load_metadata():
        logger.info("Loading metadata from APIs...")

        dto_api = DtoAPI()
        expression_api = ExpressionAPI()
        genomicfeature_api = GenomicFeatureAPI()
        # note -- exclude mitra_cc
        promotersetsig_api = PromoterSetSigAPI(
            params={"source_name": "brent_nf_cc,harbison_chip,chipexo_pugh_allevents"}
        )
        rankresponse_api = RankResponseAPI()
        regulator_api = RegulatorAPI()

        (
            dto_res,
            expression_res,
            genomicfeature_res,
            promotersetsig_res,
            rankresponse_res,
            regulator_res,
        ) = await asyncio.gather(
            dto_api.read(),
            expression_api.read(),
            genomicfeature_api.read(),
            promotersetsig_api.read(),
            rankresponse_api.read(),
            regulator_api.read(),
        )

        return {
            "dto": dto_res,
            "expression": expression_res,
            "genomicfeature": genomicfeature_res,
            "promotersetsig": promotersetsig_res,
            "rankresponse": rankresponse_res,
            "regulator": regulator_res,
        }

    # part of the async task machinery -- see load_metadata()
    @reactive.effect
    def _():
        if not load_metadata.result():
            logger.debug("Initial data is still being loaded.")
            return
        try:
            metadata_res_dict = load_metadata.result()
            logger.info("Processing metadata...")

            for key, value in metadata_res_dict.items():
                try:
                    # TODO: move this to the database serializer
                    if key == "dto":
                        parse_dto_metadata(value["metadata"], inplace=True)
                    metadata_reactives[key].set(value["metadata"])
                except KeyError:
                    logger.error(f"Failed to set metadata for {key}")
        except Exception as e:
            logger.error(f"Error processing metadata: {e}", exc_info=True)

    # Dynamic accessor for metadata reactives
    def get_metadata(key):
        """
        Factoring function to return a reactive.calc for metadata given some key
        """

        @reactive.calc
        def metadata_calc():
            return metadata_reactives[key].get()

        return metadata_calc

    # this returns a large dictionary of reactive values that are set in the
    # dataset_filters_server. One of them is the pull_data button that is used below
    # to trigger the load_data() task
    data_filters = dataset_filters_server(
        "data_filters",
        get_metadata("promotersetsig"),
        get_metadata("expression"),
        get_metadata("regulator"),
        get_metadata("rankresponse"),
        get_metadata("dto"),
    )

    # a reactive effect that triggers the load_metadata() task when the pull_data button
    # is clicked. This in terms triggers the async task -- see load_metadata()
    @reactive.effect
    @reactive.event(data_filters["pull_data"])
    def _():
        load_metadata()

    upset_plot_server(
        "upset_plot",
        data_filters["generate_plots"],
        get_metadata("promotersetsig"),
        get_metadata("expression"),
        data_filters["binding"],
        data_filters["expression"],
    )

    # based on the data filters, generate a distribution across replicates of the
    # rank response data
    rank_response_overview_plot_server(
        "rank_response_overview_plot",
        data_filters["generate_plots"],
        data_filters["rankresponse_filter"],
    )

    # generate a dto overview plot
    dto_overview_plot_server(
        "dto_overview_plot", data_filters["generate_plots"], data_filters["dto_filter"]
    )

    # generate the replicate level rank response plot and output the metadata
    # associated with those replicates
    _rr_res = rank_response_replicate_plot_server(
        "rank_response_replicate_plot",
        data_filters["rankresponse_filter"],
    )

    # use the rank response replicate metadata to generate a table
    replicate_qc_table_server("replicate_qc_table", _rr_res)


# Create an app instance
app = App(ui=app_ui, server=app_server)

if __name__ == "__main__":
    run_app(
        "yeastdnnexplorer.shiny_app.app:app", reload=True, reload_dirs=["."], port=8006
    )

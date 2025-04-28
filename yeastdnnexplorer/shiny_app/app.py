# import asyncio
# import logging
# from typing import Literal

import pandas as pd
from shiny import App, reactive, run_app, ui
from shiny.types import SilentException

# from yeastdnnexplorer.interface import (
#     DtoAPI,
#     ExpressionAPI,
#     GenomicFeatureAPI,
#     PromoterSetSigAPI,
#     RankResponseAPI,
#     RegulatorAPI,
#     UnivariateModelsAPI,
# )
# from yeastdnnexplorer.shiny_app.modules.dataset_filters.module import (
#     dataset_filters_server,
#     dataset_filters_ui,
# )

# # from yeastdnnexplorer.shiny_app.modules.dto_overview_plot import (
# #     dto_overview_plot_server,
# #     dto_overview_plot_ui,
# #     parse_dto_metadata,
# # )
# # from yeastdnnexplorer.shiny_app.modules.rank_response_overview_plot.module import (
# #     rank_response_overview_plot_server,
# #     rank_response_overview_plot_ui,
# # )
# from yeastdnnexplorer.shiny_app.modules.rank_response_replicate_plot.module import (
#     rank_response_replicate_plot_server,
#     rank_response_replicate_plot_ui,
# )
# from yeastdnnexplorer.shiny_app.modules.replicate_qc_table.module import (
#     replicate_qc_table_server,
#     replicate_qc_table_ui,
# )
# from yeastdnnexplorer.shiny_app.modules.upset_plot.module import (
#     upset_plot_server,
#     upset_plot_ui,
# )
# from yeastdnnexplorer.utils import configure_logger

# logger = logging.getLogger("shiny")

# # Call the logger configuration function
# configure_logger("shiny", level=logging.DEBUG)

# app_ui = ui.page_sidebar(
#     dataset_filters_ui("data_filters"),
#     upset_plot_ui("upset_plot"),
#     # rank_response_overview_plot_ui("rank_response_overview_plot"),
#     # dto_overview_plot_ui("dto_overview_plot"),
#     rank_response_replicate_plot_ui("rank_response_replicate_plot"),
#     replicate_qc_table_ui("replicate_qc_table"),
# )


# def app_server(input, output, session):

#     # A dictionary of reactive values to store the metadata from the APIs
#     metadata_reactives = {
#         "dto": reactive.Value(),
#         "expression": reactive.Value(),
#         "genomicfeature": reactive.Value(),
#         "promotersetsig": reactive.Value(),
#         "rankresponse": reactive.Value(),
#         "regulator": reactive.Value(),
#         "univariatemodels": reactive.Value(),
#     }

#     # this is a task that will pull the metadata from the APIs when the "pull data"
#     # (see the data filter ui) button is clicked. This follows the pattern on the
#     # python shiny async tasks example
#     # https://shiny.posit.co/py/docs/nonblocking.html
#     # without more threads on both the client and server side, and lots of bandwidth,
#     # this probably isn't any faster.
#     # NOTE: need to be careful -- this is set up with caching right now at the /export
#     # endpoints. Need to make sure filtering doesn't affect this
#     # (i expect it does 20241220)
#     @ui.bind_task_button(button_id="btn")
#     @reactive.extended_task
#     async def load_metadata():
#         with ui.Progress(min=0, max=1) as p:
#             p.set(0.5, message="Loading metadata", detail="This may take a while...")
#             logger.info("Loading metadata from APIs...")

#             dto_api = DtoAPI()
#             expression_api = ExpressionAPI()
#             genomicfeature_api = GenomicFeatureAPI()
#             # note -- exclude mitra_cc
#             promotersetsig_api = PromoterSetSigAPI(
#                 params={
#                     "source_name": "brent_nf_cc,harbison_chip,chipexo_pugh_allevents"
#                 }
#             )
#             rankresponse_api = RankResponseAPI()
#             regulator_api = RegulatorAPI()
#             univariatemodels_api = UnivariateModelsAPI()

#             (
#                 dto_res,
#                 expression_res,
#                 genomicfeature_res,
#                 promotersetsig_res,
#                 rankresponse_res,
#                 regulator_res,
#                 univariatemodels_res,
#             ) = await asyncio.gather(
#                 dto_api.read(),
#                 expression_api.read(),
#                 genomicfeature_api.read(),
#                 promotersetsig_api.read(),
#                 rankresponse_api.read(),
#                 regulator_api.read(),
#                 univariatemodels_api.read(),
#             )

#             return {
#                 "dto": dto_res,
#                 "expression": expression_res,
#                 "genomicfeature": genomicfeature_res,
#                 "promotersetsig": promotersetsig_res,
#                 "rankresponse": rankresponse_res,
#                 "regulator": regulator_res,
#                 "univariatemodels": univariatemodels_res,
#             }

#     # part of the async task machinery -- see load_metadata()
#     @reactive.effect
#     def _():
#         if not load_metadata.result():
#             logger.debug("Initial data is still being loaded.")
#             return
#         try:
#             metadata_res_dict = load_metadata.result()
#             logger.info("Processing metadata...")

#             for key, value in metadata_res_dict.items():
#                 try:
#                     metadata_reactives[key].set(value["metadata"])
#                 except KeyError:
#                     logger.error(f"Failed to set metadata for {key}")
#         except Exception as e:
#             logger.error(f"Error processing metadata: {e}", exc_info=True)

#     # Dynamic accessor for metadata reactives
#     def get_metadata(key):
#         """Factoring function to return a reactive.calc for metadata given some key."""

#         @reactive.calc
#         def metadata_calc():
#             return metadata_reactives[key].get()

#         return metadata_calc

#     # this returns a large dictionary of reactive values that are set in the
#     # dataset_filters_server. One of them is the pull_data button that is used below
#     # to trigger the load_data() task
#     data_filters = dataset_filters_server(
#         "data_filters",
#         get_metadata("promotersetsig"),
#         get_metadata("expression"),
#         get_metadata("regulator"),
#         get_metadata("rankresponse"),
#         get_metadata("dto"),
#     )

#     # a reactive effect that triggers the load_metadata() task when the pull_data button
#     # is clicked. This in terms triggers the async task -- see load_metadata()
#     @reactive.effect
#     @reactive.event(data_filters["pull_data"])
#     def _():
#         load_metadata()

#     upset_plot_server(
#         "upset_plot",
#         data_filters["generate_plots"],
#         get_metadata("promotersetsig"),
#         get_metadata("expression"),
#         data_filters["binding"],
#         data_filters["expression"],
#     )

#     # # based on the data filters, generate a distribution across replicates of the
#     # # rank response data
#     # rank_response_overview_plot_server(
#     #     "rank_response_overview_plot",
#     #     data_filters["generate_plots"],
#     #     data_filters["rankresponse_filter"],
#     # )

#     # # generate a dto overview plot
#     # dto_overview_plot_server(
#     #     "dto_overview_plot", data_filters["generate_plots"],
#     #      data_filters["dto_filter"]
#     # )

#     # generate the replicate level rank response plot and output the metadata
#     # associated with those replicates
#     _rr_res, _promotersetseg_selected = rank_response_replicate_plot_server(
#         "rank_response_replicate_plot",
#         data_filters["generate_plots"],
#         data_filters["rankresponse_filter"],
#     )

#     @reactive.calc
#     def rr_meta():
#         rr_res = _rr_res.get()
#         rr_meta = rr_res.get("metadata", pd.DataFrame())
#         pss_meta = get_metadata("promotersetsig")()
#         pss_subset = pss_meta.loc[:, ["id", "source_orig_id"]].rename(
#             columns={"id": "promotersetsig", "source_orig_id": "binding_old_id"}
#         )

#         # left join rr_meta to pss_subset by promotersetsig
#         rr_meta = rr_meta.merge(pss_subset, on="promotersetsig", how="left")
#         return rr_meta

#     def rr_meta_subset(
#         metadata_type: Literal["dto", "univariatemodels"]
#     ) -> reactive.calc:
#         """
#         Create a subset of the Rank Response metadata merged with a specific type of
#         metadata. This is intended to be used for the dto and univariate models
#         metadata. Once i decide whether to put source_orig_id onto the rr_meta, or just
#         remove that field, then this can be combined with rr_meta.

#         :param metadata_type: The type of metadata to merge with ("dto",
#             "univariatemodels", etc.).
#         :return: A reactive.calc function that computes the merged subset.

#         """

#         @reactive.calc
#         def subset():
#             logger.info(
#                 "Creating subset of Rank Response metadata with "
#                 f"{metadata_type} metadata"
#             )

#             rr_res = _rr_res.get()
#             if "metadata" not in rr_res:
#                 logger.warning("Rank Response metadata is missing 'metadata' key.")
#                 return pd.DataFrame()

#             rr_meta = rr_res.get("metadata", pd.DataFrame())
#             logger.info(f"Rank Response metadata shape: {rr_meta.shape}")

#             try:
#                 meta = get_metadata(metadata_type)()
#                 logger.info(f"{metadata_type} metadata shape: {meta.shape}")
#             except SilentException:
#                 logger.warning(f"{metadata_type} metadata not available.")
#                 return pd.DataFrame()
#             except Exception as e:
#                 logger.error(
#                     f"Error fetching {metadata_type} metadata: {e}", exc_info=True
#                 )
#                 return pd.DataFrame()

#             try:
#                 subset = rr_meta.loc[:, ["promotersetsig", "expression"]].merge(
#                     meta, on=["promotersetsig", "expression"], how="left"
#                 )
#                 logger.info(f"Subset created with shape: {subset.shape}")
#             except Exception as e:
#                 logger.error(f"Error merging metadata: {e}", exc_info=True)
#                 subset = pd.DataFrame()

#             return subset

#         return subset

#     # use the rank response replicate metadata to generate a table
#     replicate_qc_table_server(
#         "replicate_qc_table",
#         rr_meta,
#         rr_meta_subset("dto"),
#         rr_meta_subset("univariatemodels"),
#     )

app_ui = ui.page_sidebar(
    sidebar=ui.sidebar(
        ui.input_text("text", "Text input"),
        ui.input_action_button("btn", "Click me"),
    ),
)


def app_server(input, output, session):
    @reactive.effect
    def _():
        if input.text() == "hello":
            raise SilentException("Silent exception raised")

    @reactive.event(input.btn)
    def _():
        print("Button clicked")
        raise SilentException("Silent exception raised")


# Create an app instance
app = App(ui=app_ui, server=app_server)

if __name__ == "__main__":
    run_app(
        "yeastdnnexplorer.shiny_app.app:app", reload=True, reload_dirs=["."], port=8006
    )

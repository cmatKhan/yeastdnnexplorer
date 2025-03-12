import logging

from shiny import Inputs, Outputs, Session, module, reactive, render, ui
from shinywidgets import output_widget, render_plotly

from yeastdnnexplorer.interface import RankResponseAPI

from .utils import create_rank_response_replicate_plot, prepare_rank_response_data

logger = logging.getLogger("shiny")


@module.ui
def rank_response_replicate_plot_ui():
    return ui.div(
        ui.input_select("regulator", "Regulator", choices=[]),
        ui.output_ui("dynamic_expression_containers"),
    )


@module.server
def rank_response_replicate_plot_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    _trigger: reactive.value,
    _rankresponse_filtered: reactive.calc,
):
    _plot_dict_by_source = reactive.Value({})

    _rr_res = reactive.Value()

    _promotersetsig_selected = reactive.Value(set())

    @reactive.calc
    def regulators():
        logger.info("Fetching regulators for RankResponse data.")
        rr_filtered_meta = _rankresponse_filtered()
        regulator_list = rr_filtered_meta.regulator_symbol.unique().tolist()
        regulator_list.sort()
        logger.debug(f"Regulators fetched: {regulator_list}")
        return regulator_list

    @reactive.effect
    @reactive.event(_trigger)
    def _():
        regulator_list = regulators()
        ui.update_select(
            "regulator",
            choices=regulator_list,
            selected="",
        )

    # Fetch data asynchronously -- see the main app for documentation on this pattern
    # of async fetching
    @reactive.extended_task
    async def fetch_data(regulator):
        with ui.Progress(min=0, max=1) as p:
            p.set(
                0.5,
                message="Pulling RankResponse data",
                detail="This may take a while...",
            )
            regulator_api = RankResponseAPI(
                params={
                    "regulator_symbol": regulator,
                    "expression_conditions": (
                        "expression_source=kemmeren_tfko;"
                        "expression_source=mcisaac_oe,time=15"
                    ),
                }
            )
            logger.info(
                "Fetching data from RankResponseAPI with params: "
                f"{regulator_api.params}"
            )
            result = await regulator_api.read(retrieve_files=True)
            logger.info("Async fetch completed.")
            return result

    # Trigger fetch manually via button click
    @reactive.effect()
    def _():
        if not fetch_data.result():
            logger.warning("No RankResponse data fetched.")
            return
        try:
            result = fetch_data.result()
            _rr_res.set(result)
        except Exception as exc:
            logger.error(f"Error setting RankResponse data: {exc}")

    @reactive.effect
    @reactive.event(input["regulator"])
    def _():
        fetch_data(input["regulator"].get())

    # Process fetched data into plot dictionary
    @reactive.calc
    def update_plot_dict():
        rr_dict = _rr_res.get()
        if not rr_dict:
            logger.warning("No data retrieved for plots.")
            return {}

        metadata = rr_dict.get("metadata")
        expression_sources = metadata["expression_source"].unique()

        plot_dict_by_source = {}
        promotersetsig_set = set()
        for source in expression_sources:
            filtered_metadata = metadata[metadata["expression_source"] == source]
            try:
                promotersetsig_set.update(
                    [str(x) for x in filtered_metadata.promotersetsig.unique().tolist()]
                )
            except AttributeError:
                logger.warning(
                    f"Expression source {source} has no promotersetsig data."
                )
            plot_dict_by_source[source] = prepare_rank_response_data(
                {"metadata": filtered_metadata, "data": rr_dict.get("data")}
            )

        _plot_dict_by_source.set(plot_dict_by_source)

        # add random to the list so that it is initially visible
        promotersetsig_set.add("Random")
        # update the selected promotersetsig
        with reactive.isolate():
            _promotersetsig_selected.set(promotersetsig_set)

        return plot_dict_by_source

    # Prepare dynamic UI
    @reactive.Calc
    def prepare_dynamic_ui():
        plots_by_source = update_plot_dict()
        if not plots_by_source:
            return []

        containers: dict = {}
        for source, plots_dict in plots_by_source.items():
            container = ui.card(
                ui.h3(f"Expression Source: {source}"),
                *[
                    output_widget(f"plot_{source}_{expression_id}")
                    for expression_id in plots_dict.keys()
                ],
            )
            containers.setdefault(source, []).append(container)

        logger.info("Dynamic UI prepared.")
        return containers

    # Render dynamic UI
    @output
    @render.ui
    def dynamic_expression_containers():
        containers = prepare_dynamic_ui()
        if not containers:
            return ui.p("No data available.")

        # Ensure each card is correctly wrapped in a column
        ui_element = ui.row(
            *[
                ui.column(
                    6, card
                )  # Ensure the width is valid and card is correctly passed
                for container in containers.values()
                for card in container  # Flatten containers to individual cards
            ]
        )
        return ui_element

    # Render plots dynamically
    @reactive.effect()
    # @reactive.event(_plot_dict_by_source)
    def _():
        # plots_by_source = _plot_dict_by_source.get()
        plots_by_source = update_plot_dict()
        if not plots_by_source:
            logger.warning("No rank response replicate plots to render.")
            return

        for source, plots_dict in plots_by_source.items():
            for expression_id, fig in create_rank_response_replicate_plot(
                plots_dict
            ).items():
                plot_id = f"plot_{source}_{expression_id}"
                # see note below
                # promotersetsig_selected = _promotersetsig_selected.get()
                for trace in fig["data"]:
                    trace["visible"] = True
                    # NOTE: this was here b/c I wanted there to be a way to
                    # select which traces to show based on other reactives, ie
                    # promotersetsig_selected. But, adding it means that the
                    # CI aren't shown for some reason
                    # (
                    #     "legendonly"
                    #     if trace["name"] not in promotersetsig_selected
                    #     else True
                    # )

                @output(id=plot_id)
                @render_plotly
                def render_plot(fig=fig):
                    return fig

        logger.info("Rank response plots rendered successfully.")

    return _rr_res, _promotersetsig_selected

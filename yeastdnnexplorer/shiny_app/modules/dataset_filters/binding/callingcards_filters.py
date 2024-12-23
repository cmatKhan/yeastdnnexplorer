import logging
from typing import Any

from shiny import module, ui

logger = logging.getLogger("shiny")


@module.ui
def callingcards_filter_ui(row_id="callingcards_filters_rows"):
    return ui.row(
        ui.card(
            ui.card_header("Calling Cards Filters", class_="filter_card_header"),
            ui.card(
                ui.card_header("Aggregate Replicates", class_="filter_card_header"),
                ui.p(
                    "Where there are multiple passing replicates for a given condition, "
                    + "select whether to include the aggregate of those replicates"
                ),
                ui.input_checkbox_group(
                    "combined_replicates",
                    "",
                    choices=["single", "combined"],
                    selected=["single", "combined"],
                    inline=True,
                ),
                style="border: 1px solid #ccc; margin-bottom: 15px;",
            ),
            ui.card(
                ui.card_header("Data Usability", class_="filter_card_header"),
                ui.p(
                    "A value of true indicates a replicate passing automated and manual QC"
                ),
                ui.input_checkbox_group(
                    "data_usable",
                    "Data Usable",
                    choices=["pass", "fail", "unreviewed"],
                    selected=["pass", "fail", "unreviewed"],
                    inline=True,
                ),
                style="border: 1px solid #ccc; margin-bottom: 15px;",
            ),
            ui.card(
                ui.card_header("Deduplicate", class_="filter_card_header"),
                ui.p(
                    "When this is selected, if there is a aggregate replicate for a "
                    + "regulator, it will be returned instead of the individual replicates"
                ),
                ui.input_switch("deduplicate", "Deduplicate", False),
                style="border: 1px solid #ccc; margin-bottom: 15px;",
            ),
        ),
        id=row_id,
    )


@module.server
def callingcards_filter_server(input: Any, output: Any, session: Any):
    logger.debug("callingcards_filter server")
    return input.combined_replicates, input.data_usable, input.deduplicate

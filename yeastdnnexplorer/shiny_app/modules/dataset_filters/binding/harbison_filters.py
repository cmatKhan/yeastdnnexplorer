import logging
from typing import Any

from shiny import module, ui

logger = logging.getLogger("shiny")


@module.ui
def harbison_filter_ui(harbison_condition_opts=None, row_id="harbison_filters_row"):
    return ui.row(
        ui.card(
            ui.card_header("Harbison ChIP Filters", class_="filter_card_header"),
            ui.card(
                ui.card_header(
                    "Select Condition(s) -- note this only affects the Harbison data"
                ),
                ui.input_checkbox_group(
                    "harbison_conditions",
                    "Harbison Conditions",
                    choices=harbison_condition_opts() or [],
                    selected=harbison_condition_opts() or [],
                    inline=True,
                ),
                style="border: 1px solid #ccc; margin-bottom: 15px;",
            ),
        ),
        id=row_id,
    )


@module.server
def harbison_filter_server(
    input: Any,
    output: Any,
    session: Any,
):
    return input.harbison_conditions

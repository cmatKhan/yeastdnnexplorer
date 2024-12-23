import logging

from shiny import Inputs, Outputs, Session, module, reactive, ui

logger = logging.getLogger("shiny")


@module.ui
def mcisaac_filter_ui(
    row_id="callingcards_filters_rows",
    mechanism_options: list[str] | None = None,
    restriction_options: list[str] | None = None,
    time_options: list[str] | None = None,
    replicate_options: list[str] | None = None,
):
    return ui.row(
        ui.card(
            ui.card_header("McIsaac Filters", class_="filter_card_header"),
            ui.card(
                ui.card_header("Mechanism Selection", class_="filter_card_header"),
                ui.input_checkbox_group(
                    "mechanism",
                    "",
                    choices=mechanism_options(),
                    selected=mechanism_options(),
                    inline=True,
                ),
                style="border: 1px solid #ccc; margin-bottom: 15px;",
            ),
            ui.card(
                ui.card_header(
                    "Restriction Enzyme Selection", class_="filter_card_header"
                ),
                ui.input_checkbox_group(
                    "restriction",
                    "",
                    choices=restriction_options(),
                    selected=restriction_options(),
                    inline=True,
                ),
                style="border: 1px solid #ccc; margin-bottom: 15px;",
            ),
            ui.card(
                ui.card_header("Time Selection", class_="filter_card_header"),
                ui.input_checkbox_group(
                    "time",
                    "",
                    choices=time_options(),
                    selected=[x for x in time_options() if x == 15.0],
                    inline=True,
                ),
                style="border: 1px solid #ccc; margin-bottom: 15px;",
            ),
            ui.card(
                ui.card_header("Replicate Selection", class_="filter_card_header"),
                ui.input_checkbox_group(
                    "replicate",
                    "",
                    choices=replicate_options(),
                    selected=replicate_options(),
                    inline=True,
                ),
                style="border: 1px solid #ccc; margin-bottom: 15px;",
            ),
            ui.card(
                ui.card_header("Preferred Replicate", class_="filter_card_header"),
                ui.input_checkbox_group(
                    "preferred_replicate",
                    "Preferred Replicate",
                    ["true", "false"],
                ),
                style="border: 1px solid #ccc; margin-bottom: 15px;",
            ),
        ),
        id=row_id,
    )


@module.server
def mcisaac_filter_server(input: Inputs, output: Outputs, session: Session) -> tuple[
    reactive.Value[list[str]],
    reactive.Value[str],
    reactive.Value[list[str]],
    reactive.Value[list[str]],
    reactive.Value[list[str]],
    reactive.Value[str],
]:
    return (
        input.mechanism,
        input.effect_colname,
        input.restriction,
        input.time,
        input.replicate,
        input.preferred_replicate,
    )

import logging

from shiny import Inputs, Outputs, Session, module, reactive, ui

logger = logging.getLogger("shiny")


@module.ui
def tfko_filter_ui(
    row_id="tfko_filters_row",
):
    return ui.row(
        ui.card(
            ui.card_header("TFKO options", class_="filter_card_header"),
            ui.card(
                ui.card_header("Source Selection", class_="filter_card_header"),
                ui.input_checkbox_group(
                    "source",
                    "",
                    choices=["kemmeren", "hu_reimann"],
                    inline=True,
                ),
                style="border: 1px solid #ccc; margin-bottom: 15px;",
            ),
            ui.card(
                ui.card_header("Replicate Selection", class_="filter_card_header"),
                ui.input_checkbox_group(
                    "replicate",
                    "",
                    choices=["1", "2"],
                    inline=True,
                ),
                ui.card_footer(
                    "hu_reimann has only 1 replicate. " + "Some kemmeren Tfs have 2",
                    class_="filter_card_footer",
                ),
                style="border: 1px solid #ccc; margin-bottom: 15px;",
            ),
            ui.card(
                ui.card_header("Passing Replicate", class_="filter_card_header"),
                ui.input_checkbox_group(
                    "passing_replicate", "Passing Replicate", ["true", "false"]
                ),
                style="border: 1px solid #ccc; margin-bottom: 15px;",
            ),
        ),
        id=row_id,
    )


@module.server
def tfko_filter_server(
    input: Inputs, output: Outputs, session: Session
) -> tuple[reactive.Value, reactive.Value, reactive.Value]:
    return (input.source, input.replicate, input.passing_replicate)

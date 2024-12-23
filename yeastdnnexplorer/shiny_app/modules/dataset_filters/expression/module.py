import logging

from shiny import Inputs, Outputs, Session, module, reactive, ui

from yeastdnnexplorer.shiny_app.modules.dataset_filters.expression import (
    mcisaac_filters,
    tfko_filters,
)
from yeastdnnexplorer.shiny_app.modules.dataset_filters.utils import (
    create_dynamic_filter_ui,
)

logger = logging.getLogger("shiny")


@module.ui
def dataset_selector_ui():
    return ui.column(
        12,
        ui.card(
            ui.card_header("Expression Assay", class_="filter_card_header"),
            ui.card_body(
                ui.input_select(
                    "expression_assay",
                    "",
                    multiple=True,
                    choices=[],
                )
            ),
            style="border: 1px solid #ccc; margin-bottom: 15px;",
        ),
        id="expression_filters_top",
    )


@module.server
def dataset_selector_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    expression_assay_options: reactive.Value[list[str]],
    mcisaac_mechanism_options: reactive.Value[list[str]],
    mcisaac_restriction_options: reactive.Value[list[str]],
    mcisaac_time_options: reactive.Value[list[str]],
    mcisaac_replicate_options: reactive.Value[list[str]],
) -> dict[str, reactive.Value | dict[str, reactive.Value]]:
    # Define the filter element and row ids
    mcisaac_filter_element_id = "mcisaac_filter"
    mcisaac_filter_row_id = "mcisaac_filters_row"
    tfko_filter_element_id = "tfko_filter"
    tfko_filter_row_id = "tfko_filters_row"

    # this stores a list of the currently displayed assays. When an assay is selected/
    # unselected, this set is adjusted to control what filtering options are
    # displayed in the ui
    displayed_assay_current_state = reactive.Value(set())

    # server for the calling cards and chip/harbison filtering options.
    # Note that the ui is added/removed based on the selected assays

    (
        mcisaac_mechanism,
        mcisaac_effect_colname,
        mcisaac_restriction,
        mcisaac_time,
        mcisaac_replicate,
        mcisaac_preferred_replicate,
    ) = mcisaac_filters.mcisaac_filter_server(mcisaac_filter_element_id)

    tfko_source, tfko_replicate, tfko_preferred_replicate = (
        tfko_filters.tfko_filter_server(tfko_filter_element_id)
    )

    @reactive.effect()
    async def _():
        """Update the Expression assays select input with the available options."""
        logger.debug(f"Updating expression_assay choices: {expression_assay_options()}")
        ui.update_select("expression_assay", choices=expression_assay_options())

    create_dynamic_filter_ui(
        input.expression_assay,
        "overexpression",
        "expression_filters_top",
        mcisaac_filters.mcisaac_filter_ui,
        mcisaac_filter_element_id,
        mcisaac_filter_row_id,
        displayed_assay_current_state,
        mechanism_options=mcisaac_mechanism_options,
        restriction_options=mcisaac_restriction_options,
        time_options=mcisaac_time_options,
        replicate_options=mcisaac_replicate_options,
    )

    create_dynamic_filter_ui(
        input.expression_assay,
        "tfko",
        "expression_filters_top",
        tfko_filters.tfko_filter_ui,
        tfko_filter_element_id,
        tfko_filter_row_id,
        displayed_assay_current_state,
    )

    # Define a mapping of reactive outputs to dictionary keys
    return {
        "assay": input.expression_assay,
        "mcisaac": {
            "mechanism": mcisaac_mechanism,
            "effect_colname": mcisaac_effect_colname,
            "restriction": mcisaac_restriction,
            "time": mcisaac_time,
            "replicate": mcisaac_replicate,
            "preferred_replicate": mcisaac_preferred_replicate,
        },
        "tfko": {
            "source": tfko_source,
            "replicate": tfko_replicate,
            "preferred_replicate": tfko_preferred_replicate,
        },
    }

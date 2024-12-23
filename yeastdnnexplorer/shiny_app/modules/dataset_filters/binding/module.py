import logging

from shiny import Inputs, Outputs, Session, module, reactive, ui

from yeastdnnexplorer.shiny_app.modules.dataset_filters.binding import (
    callingcards_filters,
    harbison_filters,
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
            ui.input_select(
                "binding_assay",
                "Binding Dataset",
                multiple=True,
                choices=[],
            ),
            style="border: 1px solid #ccc; margin-bottom: 15px;",
        ),
        id="binding_filters_top",
    )


@module.server
def dataset_selector_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    binding_assay_options: reactive.Value[list[str]],
    harbison_conditions_options: reactive.Value[list[str]],
) -> dict[str, reactive.Value | dict[str, reactive.Value]]:
    # this stores a list of the currently displayed assays. When an assay is selected/
    # unselected, this set is adjusted to control what filtering options are
    # displayed in the ui
    displayed_assay_current_state = reactive.Value(set())

    callingcards_filter_element_id = "callingcards_filter"
    callingcards_filter_row_id = "callingcards_filters_row"
    harbison_filter_element_id = "harbison_filter"
    harbison_filter_row_id = "harbison_filters_row"

    # server for the calling cards and chip/harbison filtering options.
    # Note that the ui is added/removed based on the selected assays
    (
        callingcards_combined_replicates,
        callingcards_data_usable,
        callingcards_deduplicate,
    ) = callingcards_filters.callingcards_filter_server(callingcards_filter_element_id)

    harbison_conditions = harbison_filters.harbison_filter_server(
        harbison_filter_element_id
    )

    @reactive.effect()
    async def _():
        """Update the Binding assays select input with the available options."""
        logger.debug(f"Updating binding_assay choices: {binding_assay_options()}")
        ui.update_select("binding_assay", choices=binding_assay_options())

    create_dynamic_filter_ui(
        input.binding_assay,
        "callingcards",
        "binding_filters_top",
        callingcards_filters.callingcards_filter_ui,
        callingcards_filter_element_id,
        callingcards_filter_row_id,
        displayed_assay_current_state,
    )

    create_dynamic_filter_ui(
        input.binding_assay,
        "chip",
        "binding_filters_top",
        harbison_filters.harbison_filter_ui,
        harbison_filter_element_id,
        harbison_filter_row_id,
        displayed_assay_current_state,
        harbison_condition_opts=harbison_conditions_options,
    )

    return {
        "assay": input.binding_assay,
        "callingcards": {
            "combined_replicates": callingcards_combined_replicates,
            "data_usable": callingcards_data_usable,
            "deduplicate": callingcards_deduplicate,
        },
        "harbison": {
            "conditions": harbison_conditions,
        },
        "chipexo": {},
    }

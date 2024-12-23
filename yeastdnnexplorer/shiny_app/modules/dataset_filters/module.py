import logging
from typing import Literal

import pandas as pd
from shiny import Inputs, Outputs, Session, module, reactive, ui
from shiny.types import SilentException

from yeastdnnexplorer.shiny_app.modules.dataset_filters.binding import (
    module as binding_module,
)
from yeastdnnexplorer.shiny_app.modules.dataset_filters.expression import (
    module as expression_module,
)

logger = logging.getLogger("shiny")


@module.ui
def dataset_filters_ui():
    return ui.sidebar(
        ui.input_task_button("pull_metadata", "Pull Metadata"),
        ui.input_task_button("generate_plots", "Generate Plots"),
        binding_module.dataset_selector_ui("binding_data_filters"),
        expression_module.dataset_selector_ui("expression_data_filters"),
        width=500,
    )


@module.server
def dataset_filters_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    _promotersetsig_meta: reactive.Value,
    _expression_meta: reactive.Value,
    _regulator_meta: reactive.Value,
    _rankresponse_meta: reactive.Value,
    _dto_meta: reactive.Value,
) -> dict[str, reactive.Value | dict[str, reactive.Value]]:
    """
    This server initializes filters for binding and expression datasets and returns
    reactives for external usage.
    """

    # Define reactivity for binding metadata
    @reactive.calc
    def binding_assay_options():
        promotersetsig_meta_df = _promotersetsig_meta()
        return promotersetsig_meta_df.assay.unique().tolist()

    @reactive.calc
    def harbison_conditions_options():
        promotersetsig_meta_df = _promotersetsig_meta()
        return (
            promotersetsig_meta_df[
                promotersetsig_meta_df.source_name == "harbison_chip"
            ]
            .condition.unique()
            .tolist()
        )

    # Define reactivity for expression metadata
    @reactive.calc
    def expression_assay_options():
        expression_meta_df = _expression_meta()
        return expression_meta_df.assay.unique().tolist()

    @reactive.calc
    def mcisaac_mechanism_options():
        expression_meta_df = _expression_meta()
        return expression_meta_df.mechanism.unique().tolist()

    @reactive.calc
    def mcisaac_restriction_options():
        expression_meta_df = _expression_meta()
        return expression_meta_df.restriction.unique().tolist()

    @reactive.calc
    def mcisaac_time_options():
        expression_meta_df = _expression_meta()
        time_options = expression_meta_df.time.unique().tolist()
        time_options.sort()
        return time_options

    @reactive.calc
    def mcisaac_replicate_options():
        expression_meta_df = _expression_meta()
        return expression_meta_df.replicate.unique().tolist()

    # Call expression module and return its reactives
    expression_reactives = expression_module.dataset_selector_server(
        "expression_data_filters",
        expression_assay_options=expression_assay_options,
        mcisaac_mechanism_options=mcisaac_mechanism_options,
        mcisaac_restriction_options=mcisaac_restriction_options,
        mcisaac_time_options=mcisaac_time_options,
        mcisaac_replicate_options=mcisaac_replicate_options,
    )

    # Call binding module and return its reactives
    binding_reactives = binding_module.dataset_selector_server(
        "binding_data_filters",
        binding_assay_options=binding_assay_options,
        harbison_conditions_options=harbison_conditions_options,
    )

    # NOTE: see the following docs on SilentException for reactive.value
    # https://shiny.posit.co/py/api/express/reactive.value.html#raises
    # This is handled this way because, when a certain assay isn't selected in the UI,
    # the corresponding reactive value isn't set and it raises a SilentException and
    # would not return without the handling
    @reactive.calc
    def promotersetsig_filter():
        """
        Filter the promotersetsig metadata based on the selected filters.
        """
        df = _promotersetsig_meta()

        # Filter based on binding filters
        binding_assays = binding_reactives["assay"].get()
        if binding_assays:
            df = df[df.assay.isin(binding_assays)]

        try:
            callingcards_combined_replicates = binding_reactives["callingcards"][
                "combined_replicates"
            ].get()
            if "single" not in callingcards_combined_replicates:
                # remove records from df where assay is callingcards and single_binding
                # is null
                df = df[~((df.assay == "callingcards") & (df.single_binding.isnull()))]
            if "combined" not in callingcards_combined_replicates:
                # remove records from df where assay is callingcards and composite_binding
                # is not null
                df = df[
                    ~((df.assay == "callingcards") & (df.composite_binding.notnull()))
                ]

            callingcards_data_usable = binding_reactives["callingcards"][
                "data_usable"
            ].get()

            if callingcards_data_usable:
                # consider only the records where the assay is callingcards and filter for
                # rows in callingcards_data_usable
                df = df[
                    (df.assay == "callingcards")
                    & (df.data_usable.isin(callingcards_data_usable))
                ]

            callingcards_deduplicate = binding_reactives["callingcards"][
                "deduplicate"
            ].get()

            if callingcards_deduplicate:
                # group by regulator and source_name. Where there are multiple records, if
                # one of those records has composite_binding not null, then keep only that
                # else, keep all the records
                df = (
                    df.groupby(["regulator_symbol", "source_name"], group_keys=False)
                    .apply(
                        lambda x: (
                            x[x.composite_binding.notnull()]
                            if x.composite_binding.notnull().any()
                            else x
                        )
                    )
                    .reset_index(drop=True)
                )
        except SilentException:
            logger.debug(
                "callingcards isn't selected -- no callingcards filters present. skipping."
            )

        try:
            harbison_conditions_options = binding_reactives["harbison"][
                "conditions"
            ].get()

            if harbison_conditions_options:
                df = df[
                    (
                        (df.source_name == "harbison_chip")
                        & df.condition.isin(harbison_conditions_options)
                    )
                    | (df.source_name != "harbison_chip")
                ]
        except SilentException:
            logger.debug(
                "chip isn't selected -- no harbison conditions present. skipping."
            )

        return df

    # NOTE: see the following docs on SilentException for reactive.value
    # https://shiny.posit.co/py/api/express/reactive.value.html#raises
    # This is handled this way because, when a certain assay isn't selected in the UI,
    # the corresponding reactive value isn't set and it raises a SilentException and
    # would not return without the handling
    @reactive.calc
    def expression_filter():
        """
        Filter the expression metadata based on the selected filters.
        """
        df = _expression_meta()

        try:
            # Filter based on expression filters
            assay = expression_reactives["assay"].get()
            if assay:
                df = df[df.assay.isin(assay)]

            mcisaac_keys = [
                "mechanism",
                "restriction",
                "time",
                "preferred_replicate",
            ]
            for key in mcisaac_keys:
                mcisaac_value = expression_reactives["mcisaac"][key].get()
                if mcisaac_value:
                    # TODO: fix this hack on setting preferred_replicate to boolean
                    if key == "preferred_replicate":
                        mcisaac_value = [x.lower() == "true" for x in mcisaac_value]
                    # TODO: fix this hack on setting time (or at least make more apparent)
                    if key == "time":
                        mcisaac_value = [float(x) for x in mcisaac_value]
                    df = df[
                        ((df.source_name == "mcisaac_oe") & df[key].isin(mcisaac_value))
                        | (df.source_name != "mcisaac_oe")
                    ]

        except SilentException:
            logger.debug(
                "expression isn't selected -- no expression filters present. skipping."
            )

        try:
            tfko_source = expression_reactives["tfko"]["source"].get()
            if tfko_source:
                source_lookup = {
                    "kemmeren": "kemmeren_tfko",
                    "hu_reimann": "hu_reimann_tfko",
                }
                tfko_keys = ["replicate", "preferred_replicate"]
                try:
                    for key in tfko_keys:
                        tfko_filter_value = expression_reactives["tfko"][key].get()
                        if tfko_filter_value:

                            # TODO: fix this hack on setting preferred_replicate to boolean
                            if key == "preferred_replicate":
                                tfko_filter_value = [
                                    x.lower() == "true" for x in tfko_filter_value
                                ]
                            df = df[
                                (
                                    (df.source_name == source_lookup[tfko_source])
                                    & df[key].isin(tfko_filter_value)
                                )
                                | (df.source_name != source_lookup[tfko_source])
                            ]
                except SilentException:
                    logger.debug(
                        "tfko isn't selected -- no tfko filters present. skipping."
                    )
        except SilentException:
            logger.debug("tfko isn't selected -- no tfko filters present. skipping.")

        return df

    def dto_rr_filter(which_meta: Literal["rankresponse", "dto"]):
        """
        Factory function to filter the rankresponse or dto metadata
        based on the selected filters.

        :param which_meta: str, either "rankresponse" or "dto"
        """

        @reactive.calc
        def inner():
            meta_df = _dto_meta() if which_meta == "dto" else _rankresponse_meta()
            # raise error if the fields `promotersetsig` and `expression` are not present
            if not {"promotersetsig", "expression"}.issubset(meta_df.columns):
                raise ValueError(
                    "The metadata dataframe should have columns 'promotersetsig' and 'expression'"
                )
            promotersetsig_filtered = promotersetsig_filter()
            expression_filtered = expression_filter()

            # filter the rankresponse table based on the promotersetsig and experssion fitlers

            df = meta_df[
                meta_df.promotersetsig.isin(promotersetsig_filtered.id)
                & meta_df.expression.isin(expression_filtered.id)
            ]

            return df

        return inner

    # Return all reactives
    return {
        "pull_data": input.pull_metadata,
        "generate_plots": input.generate_plots,
        "binding": binding_reactives,
        "expression": expression_reactives,
        "promotersetsig_filter": promotersetsig_filter,
        "expression_filter": expression_filter,
        "rankresponse_filter": dto_rr_filter("rankresponse"),
        "dto_filter": dto_rr_filter("dto"),
    }

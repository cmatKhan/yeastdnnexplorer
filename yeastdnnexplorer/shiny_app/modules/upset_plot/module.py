import json
import logging

from shiny import Inputs, Outputs, Session, module, reactive, req, ui
from shinywidgets import output_widget, render_widget
from upsetjs_jupyter_widget import UpSetJSWidget

logger = logging.getLogger("shiny")


@module.ui
def upset_plot_ui():
    return ui.page_fillable(
        ui.tags.head(
            ui.tags.script(
                """
                $(document).on("shiny:connected", function() {
                    $(window).resize(function() {
                        var w = $(this).width();
                        var h = $(this).height();
                        Shiny.setInputValue("pltChange", {width: w, height: h});
                    });
                });
                """
            )
        ),
        ui.card(
            output_widget("upsetjs_plot"),
        ),
    )


@module.server
def upset_plot_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    trigger: reactive.Value,
    _promotersetsig_meta: reactive.Calc,
    _expression_meta: reactive.Calc,
    binding_reactives: dict[str, reactive.Value | dict[str, reactive.Value]],
    expression_reactives: dict[str, reactive.Value | dict[str, reactive.Value]],
) -> None:
    """Server logic for the UpSet plot module."""

    @reactive.calc
    def regulators_by_source():
        """Create a dictionary where the keys are the binding sources, and the values
        are lists of unique regulator symbols for each source."""
        source_regulator_dict = {}
        promotersetsig_meta = _promotersetsig_meta()
        source_regulator_dict.update(
            {
                k: promotersetsig_meta[promotersetsig_meta["source_name"] == k][
                    "regulator_symbol"
                ]
                .unique()
                .tolist()
                for k in promotersetsig_meta.source_name.unique().tolist()
            }
        )

        expression_meta = _expression_meta()
        source_regulator_dict.update(
            {
                k: expression_meta[expression_meta["source_name"] == k][
                    "regulator_symbol"
                ]
                .unique()
                .tolist()
                for k in expression_meta.source_name.unique().tolist()
            }
        )

        return source_regulator_dict

    @reactive.calc
    def selected_dataset():
        """Derive the selected dataset based on binding and expression assays."""
        binding_assays = binding_reactives["assay"].get()  # type: ignore
        expression_assays = expression_reactives["assay"].get()  # type: ignore

        data_sets = set()

        if "callingcards" in binding_assays:
            data_sets.add("callingcards")
        if "chipexo" in binding_assays:
            data_sets.add("chipexo")
        if "chip" in binding_assays:
            data_sets.add("harbison_chip")

        if "tfko" in expression_assays:
            tfko_source = expression_reactives["tfko"]["source"].get()
            if "kemmeren" in tfko_source:
                data_sets.add("kemmeren_tfko")
            if "hu_reimann" in tfko_source:
                data_sets.add("hu_reimann_tfko")

        if "overexpression" in expression_assays:
            data_sets.add("mcisaac_oe")

        logger.info(f"Selected datasets: {data_sets}")
        return data_sets

    @reactive.calc
    def upset_combinations_list():
        """Derive the combinations list from the UpSetJS widget."""
        w = upset_plot_object()
        combinations_list = []
        for combination in w.combinations:
            combinations_list.append({x.name.strip() for x in combination.sets})
        return combinations_list

    upset_plot_object = reactive.Value()

    @reactive.effect
    @reactive.event(trigger)
    def _():
        """Update the UpSetJS plot when the trigger event occurs."""
        w = UpSetJSWidget[str]()
        regulators_by_source_dict = regulators_by_source()

        w.from_dict(
            {key: regulators_by_source_dict[key] for key in regulators_by_source_dict},
            order_by="name",
        )
        w.generate_intersections(order_by="degree", min_degree=2, empty=True)
        w.mode = "click"
        w.title = "Chart Title"
        w.description = "A long chart description"
        w.width = "100%"
        w.height = "100%"

        # React to selection changes
        def selection_changed(s):
            regulators = s.elems if s else None
            logger.info(f"Selection changed: {regulators}")

        w.on_selection_changed(selection_changed)
        upset_plot_object.set(w)

    @reactive.effect
    @reactive.event(input.pltChange)
    def _():
        """
        This is an effort to resize the plot when the window is resized.

        It isn't working yet

        """
        req(input.pltChange)
        try:
            resize_info = json.loads(input.pltChange())
            logger.info(
                f"Resizing plot to width: {resize_info['width']}, "
                f"height: {resize_info['height']}"
            )
            w = upset_plot_object.get()
            w.width = f"{resize_info['width']}px"
            w.height = f"{resize_info['height'] - 100}px"
            upset_plot_object.set(w)
        except Exception as e:
            logger.error(f"Error handling resize: {e}")

    @render_widget()
    def upsetjs_plot():
        return upset_plot_object.get()

    # Adjust selection dynamically based on `selected_dataset`
    @reactive.effect
    @reactive.event(selected_dataset)
    def update_selection():
        req(upset_combinations_list())
        w = upset_plot_object.get()
        selected = selected_dataset()
        for combination in upset_combinations_list():
            if selected == combination:
                w.selection = combination
                upset_plot_object.set(w)
                break

    session.on_ended(on_shutdown)


def on_shutdown():
    logger.debug("Shutting down...")

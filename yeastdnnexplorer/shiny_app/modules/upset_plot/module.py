# import json
# import logging

# from shiny import Inputs, Outputs, Session, module, reactive, req, ui
# from shinywidgets import output_widget, render_widget
# from upsetjs_jupyter_widget import UpSetJSWidget, UpSetSetCombination

# logger = logging.getLogger("shiny")


# # Define the UI for the module
# @module.ui
# def upset_plot_ui():
#     return ui.page_fillable(
#         ui.tags.head(
#             ui.tags.script(
#                 """
#                 $(document).on("shiny:connected", function() {
#                     $(window).resize(function() {
#                         var w = $(this).width();
#                         var h = $(this).height();
#                         console.log("Resize event triggered:", { width: w, height: h });
#                         Shiny.setInputValue("pltChange", {width: w, height: h});
#                     });
#                 });
#                 """
#             )
#         ),
#         ui.card(
#             output_widget("upsetjs_plot"),
#         ),
#     )


# # Define the server logic for the module
# @module.server
# def upset_plot_server(
#     input: Inputs,
#     output: Outputs,
#     session: Session,
#     trigger: reactive.value,
#     _promotersetsig_meta: reactive.calc,
#     _expression_meta: reactive.calc,
#     binding_reactives: dict[str, reactive.Value | dict[str, reactive.Value]],
#     expression_reactives: dict[str, reactive.Value | dict[str, reactive.Value]],
# ) -> None:
#     """
#     Server logic for the UpSet plot module.

#     :param input: See shiny.Inputs
#     :param output: See shiny.Outputs
#     :param session: See shiny.Session
#     :param trigger: A reactive value that triggers the UpSet plot to be updated. This
#         is expected to be a button or something similar on which @reactive.event can
#         depend
#     :param _promotersetsig_meta: A reactive.calc that contains the metadata for the
#         promoter set signatures
#     :param _expression_meta: A reactive.calc that contains the metadata for the
#         expression data
#     :param binding_reactives: A dictionary containing the reactive values for the
#         binding assays after filtering. See output of data_filters server module
#     :param expression_reactives: A dictionary containing the reactive values for the
#         expression assays after filtering. See output of data_filters server module
#     """
#     _upset_reactives = reactive.Value({"sets": [], "regulators": []})
#     _upset_plot_object = reactive.Value()
#     _upset_combinations_list = reactive.Value()
#     _selected_dataset = reactive.Value()

#     @reactive.calc
#     def regulators_by_source():
#         """
#         Create a dictionary where they keys are the binding sources and the values are
#         a listo of unique regulator symbols for each source.

#         :return: A dictionary of regulator symbols by binding source.
#             Eg, {"harbison_chip": ["GAL4", "SWI4"], "callingcards": ["GAL4", "SWI4", "GCN4"]}
#         :rtype: dict
#         """
#         source_regulator_dict = {}
#         promotersetsig_meta = _promotersetsig_meta()
#         source_regulator_dict.update(
#             {
#                 k: promotersetsig_meta[promotersetsig_meta["source_name"] == k][
#                     "regulator_symbol"
#                 ]
#                 .unique()
#                 .tolist()
#                 for k in promotersetsig_meta.source_name.unique().tolist()
#             }
#         )

#         expression_meta = _expression_meta()
#         source_regulator_dict.update(
#             {
#                 k: expression_meta[expression_meta["source_name"] == k][
#                     "regulator_symbol"
#                 ]
#                 .unique()
#                 .tolist()
#                 for k in expression_meta.source_name.unique().tolist()
#             }
#         )

#         return source_regulator_dict

#     # Reactive output to display the UpSetJS plot
#     @reactive.event(trigger)
#     def _():
#         # Create the UpSetJSWidget instance
#         w = UpSetJSWidget[str]()

#         regulators_by_source_dict = regulators_by_source()

#         # Populate the widget with data
#         w.from_dict(
#             {
#                 f"{key}": regulators_by_source_dict[key]
#                 for key in regulators_by_source_dict
#             },
#             order_by="name",
#         )
#         w.generate_intersections(
#             order_by="degree",
#             min_degree=2,  # Minimum 2 sets in an intersection
#             empty=True,  # Include empty intersections
#         )

#         # Set the widget's mode to "click" to enable selection on click
#         w.mode = "click"

#         # add a title and description
#         w.title = "Chart Title"
#         w.description = "a long chart description"
#         # w.set_name = "Set Label"
#         # w.combination_name = "Combination Label"
#         w.width = "100%"
#         w.height = "100%"

#         # Define a function to capture selection changes
#         def selection_changed(s):
#             _upset_reactives["sets"].set(s.sets if s else None)
#             _upset_reactives["regulators"].set(s.elems if s else None)

#         # Attach the callback to the widget's selection change event
#         w.on_selection_changed(selection_changed)

#         combinations_list: list[set[UpSetSetCombination]] = [set()] * len(
#             w.combinations
#         )
#         for i, combination in enumerate(w.combinations):
#             combinations_list[i] = {x.name.strip() for x in combination.sets}

#         _upset_plot_object.set(w)
#         _upset_combinations_list.set(combinations_list)

#     # React to resize events
#     # TODO: This isn't working to resize the plot
#     # the resize trigger from the javascript is getting triggered 3 times each time the
#     # page resizes
#     # see https://stackoverflow.com/questions/34082452/shiny-send-an-resize-event-when-ui-element-gets-resized
#     @reactive.effect()
#     @reactive.event(input.pltChange)
#     def resize_upset_plot():
#         req(input.pltChange)
#         logger.info(f"Received pltChange: {input.pltChange()}")
#         try:
#             resize_info = json.loads(input.pltChange())
#             logger.info(
#                 f"Resizing plot to width: {resize_info['width']}, height: {resize_info['height']}"
#             )
#             w = _upset_plot_object.get()
#             w.width = f"{resize_info['width']}px"
#             w.height = f"{resize_info['height'] - 100}px"  # Subtract padding/margins
#             _upset_plot_object.set(w)
#         except Exception as e:
#             logger.error(f"Error handling resize: {e}")

#     # Render the UpSetJSWidget
#     @render_widget()
#     def upsetjs_plot():
#         return _upset_plot_object.get()

#     # Set the selected dataset
#     @reactive.effect()
#     def _():
#         binding_assays = binding_reactives["assay"].get()  # type: ignore
#         expression_assays = expression_reactives["assay"].get()  # type: ignore

#         # Initialize an empty set for selected data sets
#         data_sets = set()

#         # Check for binding assays
#         if "callingcards" in binding_assays:
#             data_sets.add("callingcards")
#         if "chipexo" in binding_assays:
#             data_sets.add("chipexo")
#         if "chip" in binding_assays:
#             data_sets.add("harbison_chip")

#         # Check for tfko in expression assays before accessing its source
#         if "tfko" in expression_assays:
#             tfko_source = expression_reactives["tfko"]["source"].get()
#             if "kemmeren" in tfko_source:
#                 data_sets.add("kemmeren_tfko")
#             if "hu_reimann" in tfko_source:
#                 data_sets.add("hu_reimann_tfko")

#         # Check for overexpression in expression assays
#         if "overexpression" in expression_assays:
#             data_sets.add("mcisaac_oe")

#         logger.info(f"Data sets: {data_sets}")

#         # Update the selected dataset reactive
#         _selected_dataset.set(data_sets)

#     # React to changes in binding and expression assay selections
#     @reactive.effect()
#     @reactive.event(_selected_dataset)
#     def _():
#         req(_upset_plot_object)
#         req(_upset_combinations_list)

#         w = _upset_plot_object.get()
#         w_new = w.copy()

#         # if multiple assays are selected from binding and/or expression, then find
#         # the index of the combination that matches and set that as the selection
#         # in the upset plot object.
#         if len(_selected_dataset.get()) < 2:
#             if w_new.selection:
#                 w_new.selection = None
#                 _upset_plot_object.set(w_new)
#         else:
#             for i, combination in enumerate(_upset_combinations_list.get()):
#                 if _selected_dataset.get() == combination:
#                     # w_new.selection = w.combinations[i]
#                     w_new.selection = w.combinations[i]
#                     logger.debug(
#                         f"Setting selection in upset plot object: {w_new.selection}"
#                     )
#                     _upset_plot_object.set(w_new)
#                     break

#     # Register the shutdown callback
#     session.on_ended(on_shutdown)


# # This is here just to show it is possible
# def on_shutdown():
#     logger.debug("Shutting down...")

import json
import logging

from shiny import Inputs, Outputs, Session, module, reactive, req, ui
from shinywidgets import output_widget, render_widget
from upsetjs_jupyter_widget import UpSetJSWidget, UpSetSetCombination

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
    """
    Server logic for the UpSet plot module.
    """

    @reactive.calc
    def regulators_by_source():
        """
        Create a dictionary where the keys are the binding sources, and the values are
        lists of unique regulator symbols for each source.
        """
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
        """
        Derive the selected dataset based on binding and expression assays.
        """
        binding_assays = binding_reactives["assay"].get()
        expression_assays = expression_reactives["assay"].get()

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
        """
        Derive the combinations list from the UpSetJS widget.
        """
        w = upset_plot_object()
        combinations_list = []
        for combination in w.combinations:
            combinations_list.append({x.name.strip() for x in combination.sets})
        return combinations_list

    upset_plot_object = reactive.Value()

    @reactive.effect
    @reactive.event(trigger)
    def _():
        """
        Update the UpSetJS plot when the trigger event occurs.
        """
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
        This is an effort to resize the plot when the window is resized. It isn't
        working yet
        """
        req(input.pltChange)
        try:
            resize_info = json.loads(input.pltChange())
            logger.info(
                f"Resizing plot to width: {resize_info['width']}, height: {resize_info['height']}"
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

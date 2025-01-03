import logging

from shiny import Inputs, Outputs, Session, module, reactive, render, req, ui
from shiny.types import SilentException

logger = logging.getLogger("shiny")


@module.ui
def replicate_qc_table_ui():
    return ui.column(
        12,
        ui.row(ui.output_data_frame("overview_table")),
        ui.row(ui.accordion(id="replicate_qc_accordion", multiple=True, open=False)),
    )


@module.server
def replicate_qc_table_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    _rr_meta: reactive.calc,
    _dto_meta: reactive.calc,
    _univariatemodels_meta: reactive.calc,
):
    height = "500px"
    width = "100%"

    _expression_sources = reactive.Value([])

    @reactive.calc
    def selected_promotersetsig():
        data_selected = overview_table.data_view(selected=True)
        req(not data_selected.empty)
        selected_pss = data_selected.promotersetsig.unique().tolist()
        logger.info(f"Selected promotersetsig: {selected_pss}")
        return selected_pss

    # Render the Overview table
    @output
    @render.data_frame
    def overview_table():
        rr_meta = _rr_meta()
        overview_df = rr_meta.loc[
            :, ["promotersetsig", "binding_source", "binding_old_id"]
        ].drop_duplicates()
        return render.DataGrid(overview_df, selection_mode="rows")

    @reactive.effect
    def _():
        logger.info("Replicate QC Table Server")
        rr_meta = _rr_meta()
        dto_meta = _dto_meta()
        univariatemodels_meta = _univariatemodels_meta()

        if rr_meta.empty:
            logger.warning("No data available in metadata.")
            return

        logger.info(
            "Rendering replicate QC tables for "
            f"{rr_meta['expression_source'].unique()} sources"
        )

        # Get unique expression sources
        expression_sources = rr_meta["expression_source"].unique()

        for source in expression_sources:
            # Subset data for the current source
            table_dict = {
                "rank_response": rr_meta[
                    rr_meta["expression_source"] == source
                ].reset_index(),
                "dto": dto_meta[dto_meta["expression_source"] == source].reset_index(),
                "univariatemodels": univariatemodels_meta[
                    univariatemodels_meta["expression_source"] == source
                ].reset_index(),
            }

            for table_name, table_df in table_dict.items():
                table_id = f"table_{table_name}_{source}"
                # drop columns uploader and modifier if they exist
                try:
                    table_df = table_df.drop(
                        columns=[
                            "uploader",
                            "upload_date",
                            "modifier",
                            "modified_date",
                            "regulator_symbol",
                            "regulator_locus_tag",
                            "expression_source",
                        ]
                    )
                except KeyError:
                    logger.warning(
                        "Columns uploader and modifier do not "
                        f"exist in {table_name} table"
                    )

                @output(id=table_id)
                @render.data_frame
                def render_table(source=source, subset_df=table_df):
                    def style_fn(df):
                        try:
                            selected_pss = selected_promotersetsig()
                            logger.debug(
                                "Selected promotersetsig in render_table: "
                                f"{selected_pss}"
                            )
                            matching_rows = df.index[
                                df["promotersetsig"].isin(selected_pss)
                            ].tolist()
                            logger.debug(
                                f"Matching rows in render_table: {matching_rows}"
                            )
                            return (
                                [
                                    {
                                        "rows": matching_rows,
                                        "style": {
                                            "backgroundColor": "lightblue",
                                            "fontWeight": "bold",
                                        },
                                    }
                                ]
                                if matching_rows
                                else []
                            )
                        except SilentException:
                            return []

                    return render.DataTable(
                        subset_df.drop(columns=["index"]),
                        width=width,
                        height=height,
                        filters=True,
                        summary=True,
                        selection_mode="rows",
                        styles=style_fn,
                    )

            # Dynamically insert the accordion panel if it doesn't already exist
            with reactive.isolate():
                current_accordion_sources = _expression_sources.get()
                if source not in current_accordion_sources:
                    tab_panels = [
                        ui.nav_panel(
                            table_name,
                            ui.output_data_frame(f"table_{table_name}_{source}"),
                        )
                        for table_name in table_dict.keys()
                    ]
                    tab_set = ui.navset_tab(*tab_panels)
                    ui.insert_accordion_panel(
                        id="replicate_qc_accordion",
                        panel=ui.accordion_panel(source, tab_set, open=False),
                    )
                    _expression_sources.set(current_accordion_sources + [source])

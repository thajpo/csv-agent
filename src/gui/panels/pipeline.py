"""
Pipeline control panel - start/stop stages, configure options.
"""

import dearpygui.dearpygui as dpg
from pathlib import Path

from src.gui.state import state
from src.gui import workers


def create_pipeline_panel(parent: int | str):
    """Create the pipeline control panel."""
    with dpg.child_window(parent=parent, width=-1, height=200, border=True):
        dpg.add_text("Pipeline Control", color=(150, 200, 255))
        dpg.add_separator()

        # Stage selector
        with dpg.group(horizontal=True):
            dpg.add_text("Stage:")
            dpg.add_combo(
                items=["Question Generation", "Episode Generation"],
                default_value="Question Generation",
                tag="pipeline_stage",
                width=200,
                callback=_on_stage_change,
            )

        dpg.add_spacer(height=5)

        # Question generation options
        with dpg.group(tag="qgen_options"):
            with dpg.group(horizontal=True):
                dpg.add_text("Mode:")
                dpg.add_radio_button(
                    items=["Synthetic", "LLM"],
                    default_value="Synthetic",
                    tag="qgen_mode",
                    horizontal=True,
                )

            with dpg.group(horizontal=True):
                dpg.add_text("Max Questions:")
                dpg.add_input_int(
                    tag="qgen_n_questions",
                    default_value=10,
                    width=100,
                    min_value=1,
                    max_value=100,
                )

            dpg.add_button(
                label="Select CSVs...",
                callback=_open_csv_selector,
                tag="csv_select_btn",
            )
            dpg.add_text("0 CSVs selected", tag="csv_count")

        # Episode generation options
        with dpg.group(tag="epgen_options", show=False):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Parallel", tag="epgen_parallel", default_value=True)
                dpg.add_text("Workers:")
                dpg.add_input_int(
                    tag="epgen_workers",
                    default_value=4,
                    width=80,
                    min_value=1,
                    max_value=16,
                )

        dpg.add_spacer(height=10)

        # Control buttons
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Start",
                callback=_start_pipeline,
                tag="start_btn",
                width=100,
            )
            dpg.add_button(
                label="Stop",
                callback=_stop_pipeline,
                tag="stop_btn",
                width=100,
                enabled=False,
            )

        # Status
        with dpg.group(horizontal=True):
            dpg.add_text("Status:")
            dpg.add_text("Idle", tag="pipeline_status", color=(150, 150, 150))


def _on_stage_change(sender, value):
    """Handle stage selector change."""
    if value == "Question Generation":
        dpg.show_item("qgen_options")
        dpg.hide_item("epgen_options")
    else:
        dpg.hide_item("qgen_options")
        dpg.show_item("epgen_options")


def _open_csv_selector(sender, data):
    """Open CSV file selector dialog."""
    # Discover CSVs
    csv_sources = workers.discover_csv_sources()

    if not dpg.does_item_exist("csv_selector_window"):
        with dpg.window(
            label="Select CSVs",
            tag="csv_selector_window",
            width=500,
            height=400,
            modal=True,
            on_close=lambda: dpg.hide_item("csv_selector_window"),
        ):
            dpg.add_text("Select datasets to process:")
            dpg.add_separator()

            with dpg.child_window(height=300, border=True):
                for csv_path in csv_sources:
                    name = csv_path.parent.name if csv_path.name == "data.csv" else csv_path.stem
                    dpg.add_checkbox(
                        label=name,
                        tag=f"csv_check_{name}",
                        user_data=str(csv_path),
                    )

            with dpg.group(horizontal=True):
                dpg.add_button(label="Select All", callback=_select_all_csvs)
                dpg.add_button(label="Clear", callback=_clear_csv_selection)
                dpg.add_button(label="OK", callback=_confirm_csv_selection)

    dpg.show_item("csv_selector_window")


def _select_all_csvs():
    """Select all CSV checkboxes."""
    for item in dpg.get_all_items():
        if dpg.get_item_alias(item).startswith("csv_check_"):
            dpg.set_value(item, True)


def _clear_csv_selection():
    """Clear all CSV checkboxes."""
    for item in dpg.get_all_items():
        if dpg.get_item_alias(item).startswith("csv_check_"):
            dpg.set_value(item, False)


def _confirm_csv_selection():
    """Confirm CSV selection and update count."""
    selected = []
    for item in dpg.get_all_items():
        alias = dpg.get_item_alias(item)
        if alias.startswith("csv_check_") and dpg.get_value(item):
            path = dpg.get_item_user_data(item)
            if path:
                selected.append(Path(path))

    state.csv_sources = selected
    dpg.set_value("csv_count", f"{len(selected)} CSVs selected")
    dpg.hide_item("csv_selector_window")


def _get_selected_csvs() -> list[Path]:
    """Get currently selected CSV paths."""
    selected = []
    for item in dpg.get_all_items():
        alias = dpg.get_item_alias(item)
        if alias.startswith("csv_check_") and dpg.get_value(item):
            path = dpg.get_item_user_data(item)
            if path:
                selected.append(Path(path))
    return selected if selected else None


def _start_pipeline(sender, data):
    """Start the selected pipeline stage."""
    stage_name = dpg.get_value("pipeline_stage")

    if stage_name == "Question Generation":
        mode = dpg.get_value("qgen_mode")
        n_questions = dpg.get_value("qgen_n_questions")
        csv_paths = _get_selected_csvs()

        if mode == "Synthetic":
            workers.start_pipeline(
                stage="synthetic_generator",
                csv_paths=csv_paths,
                n_questions=n_questions,
            )
        else:
            workers.start_pipeline(
                stage="question_gen",
                csv_paths=csv_paths,
            )
    else:
        parallel = dpg.get_value("epgen_parallel")
        n_workers = dpg.get_value("epgen_workers")

        workers.start_pipeline(
            stage="synthetic_episodes",
            parallel=parallel,
            n_workers=n_workers,
        )

    # Update UI
    dpg.disable_item("start_btn")
    dpg.enable_item("stop_btn")
    dpg.set_value("pipeline_status", "Running...")
    dpg.configure_item("pipeline_status", color=(100, 255, 100))


def _stop_pipeline(sender, data):
    """Stop the running pipeline."""
    workers.stop_pipeline()

    dpg.enable_item("start_btn")
    dpg.disable_item("stop_btn")
    dpg.set_value("pipeline_status", "Stopped")
    dpg.configure_item("pipeline_status", color=(255, 150, 100))


def update_pipeline_panel():
    """Update pipeline panel based on state."""
    status = state.pipeline.status

    if status == "running":
        dpg.disable_item("start_btn")
        dpg.enable_item("stop_btn")
        elapsed = f"Running... ({state.pipeline.elapsed:.0f}s)"
        dpg.set_value("pipeline_status", elapsed)
        dpg.configure_item("pipeline_status", color=(100, 255, 100))

    elif status == "completed":
        dpg.enable_item("start_btn")
        dpg.disable_item("stop_btn")
        dpg.set_value("pipeline_status", f"Completed ({state.pipeline.elapsed:.0f}s)")
        dpg.configure_item("pipeline_status", color=(100, 200, 255))

    elif status == "failed":
        dpg.enable_item("start_btn")
        dpg.disable_item("stop_btn")
        dpg.set_value("pipeline_status", "Failed")
        dpg.configure_item("pipeline_status", color=(255, 100, 100))

    elif status == "stopped":
        dpg.enable_item("start_btn")
        dpg.disable_item("stop_btn")
        dpg.set_value("pipeline_status", "Stopped")
        dpg.configure_item("pipeline_status", color=(255, 150, 100))

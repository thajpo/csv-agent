"""
Progress dashboard panel - progress bars, counters, logs.
"""

import dearpygui.dearpygui as dpg

from src.gui.state import state


def create_progress_panel(parent: int | str):
    """Create the progress dashboard panel."""
    with dpg.child_window(parent=parent, width=-1, height=300, border=True):
        dpg.add_text("Progress", color=(150, 200, 255))
        dpg.add_separator()

        # Summary stats
        with dpg.group(horizontal=True):
            dpg.add_text("Datasets:")
            dpg.add_text("0 / 0", tag="progress_datasets")
            dpg.add_spacer(width=30)
            dpg.add_text("Questions:")
            dpg.add_text("0 / 0", tag="progress_questions")
            dpg.add_spacer(width=30)
            dpg.add_text("Verified:")
            dpg.add_text("0", tag="progress_verified", color=(100, 255, 100))
            dpg.add_spacer(width=10)
            dpg.add_text("Failed:")
            dpg.add_text("0", tag="progress_failed", color=(255, 100, 100))

        dpg.add_spacer(height=10)

        # Per-dataset progress bars container
        with dpg.child_window(
            tag="progress_bars_container",
            height=100,
            border=True,
        ):
            dpg.add_text("No datasets processing", tag="no_datasets_text")

        dpg.add_spacer(height=10)

        # Log output
        dpg.add_text("Log Output:")
        with dpg.child_window(
            tag="log_container",
            height=-1,
            border=True,
        ):
            dpg.add_text("", tag="log_output", wrap=0)


def update_progress_panel():
    """Update progress panel with current state."""
    p = state.pipeline

    # Count totals
    total_datasets = len(p.datasets)
    done_datasets = sum(1 for d in p.datasets.values() if d.done >= d.total and d.total > 0)
    total_questions = sum(d.total for d in p.datasets.values())
    done_questions = sum(d.done for d in p.datasets.values())
    verified = sum(d.verified for d in p.datasets.values())
    failed = sum(d.failed for d in p.datasets.values())

    # Update summary
    dpg.set_value("progress_datasets", f"{done_datasets} / {total_datasets}")
    dpg.set_value("progress_questions", f"{done_questions} / {total_questions}")
    dpg.set_value("progress_verified", str(verified))
    dpg.set_value("progress_failed", str(failed))

    # Update per-dataset progress bars
    _update_progress_bars()

    # Update log
    if p.log_lines:
        # Show last 50 lines
        log_text = "\n".join(p.log_lines[-50:])
        dpg.set_value("log_output", log_text)

        # Auto-scroll if enabled
        if state.log_scroll_to_bottom:
            dpg.set_y_scroll("log_container", dpg.get_y_scroll_max("log_container"))


def _update_progress_bars():
    """Update the per-dataset progress bars."""
    p = state.pipeline

    # Delete old progress bars
    if dpg.does_item_exist("no_datasets_text"):
        if p.datasets:
            dpg.hide_item("no_datasets_text")
        else:
            dpg.show_item("no_datasets_text")

    # Create/update progress bars for each dataset
    for name, ds in p.datasets.items():
        bar_tag = f"progress_bar_{name}"
        label_tag = f"progress_label_{name}"

        if not dpg.does_item_exist(bar_tag):
            # Create new progress bar group
            with dpg.group(parent="progress_bars_container", horizontal=True, tag=f"progress_group_{name}"):
                dpg.add_text(f"{name[:20]:20}", tag=label_tag)
                dpg.add_progress_bar(
                    tag=bar_tag,
                    default_value=0.0,
                    width=200,
                    overlay=f"0/{ds.total}",
                )
                dpg.add_text("", tag=f"progress_status_{name}")

        # Update values
        progress = ds.done / ds.total if ds.total > 0 else 0.0
        dpg.set_value(bar_tag, progress)
        dpg.configure_item(bar_tag, overlay=f"{ds.done}/{ds.total}")

        # Status indicator
        if name == p.current_dataset:
            dpg.set_value(f"progress_status_{name}", " [running]")
            dpg.configure_item(f"progress_status_{name}", color=(100, 255, 100))
        elif ds.done >= ds.total and ds.total > 0:
            dpg.set_value(f"progress_status_{name}", f" [done: {ds.verified}v/{ds.failed}f]")
            dpg.configure_item(f"progress_status_{name}", color=(150, 150, 150))
        else:
            dpg.set_value(f"progress_status_{name}", " [queued]")
            dpg.configure_item(f"progress_status_{name}", color=(150, 150, 150))

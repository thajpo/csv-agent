"""GUI panels for the dashboard."""

from src.gui.panels.system import create_system_panel, update_system_panel
from src.gui.panels.pipeline import create_pipeline_panel
from src.gui.panels.progress import create_progress_panel, update_progress_panel
from src.gui.panels.explorer import create_explorer_panel
from src.gui.panels.trace import create_trace_panel, show_trace

__all__ = [
    "create_system_panel",
    "update_system_panel",
    "create_pipeline_panel",
    "create_progress_panel",
    "update_progress_panel",
    "create_explorer_panel",
    "create_trace_panel",
    "show_trace",
]

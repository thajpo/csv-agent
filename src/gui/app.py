"""
Main GUI application - window setup, layout, render loop.
"""

import time

import dearpygui.dearpygui as dpg

from src.gui.panels import system, pipeline, progress, explorer, trace
from src.gui import workers


def create_app():
    """Initialize and create the main application window."""
    dpg.create_context()
    dpg.create_viewport(title="CSV-Agent Dashboard", width=1400, height=900)

    # Configure theme
    _setup_theme()

    # Create main window
    with dpg.window(tag="main_window"):
        # Top bar - system monitor + pipeline controls
        with dpg.group(horizontal=True):
            system.create_system_panel("main_window")
            with dpg.child_window(width=-1, height=150, border=True):
                pipeline.create_pipeline_panel(dpg.last_container())

        dpg.add_spacer(height=5)

        # Progress dashboard
        progress.create_progress_panel("main_window")

        dpg.add_spacer(height=5)

        # Bottom section - explorer + trace viewer
        with dpg.group(horizontal=True):
            explorer.create_explorer_panel("main_window")
            dpg.add_spacer(width=5)
            trace.create_trace_panel("main_window")

    dpg.set_primary_window("main_window", True)


def _setup_theme():
    """Configure the application theme."""
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            # Dark background
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (30, 30, 35))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (25, 25, 30))

            # Accent colors
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (50, 80, 120))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (50, 80, 120))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (60, 100, 140))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (70, 110, 150))

            # Buttons
            dpg.add_theme_color(dpg.mvThemeCol_Button, (50, 80, 120))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 100, 140))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (70, 110, 150))

            # Frame/input backgrounds
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (40, 40, 50))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (50, 50, 60))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (60, 60, 70))

            # Scrollbar
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (20, 20, 25))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (50, 50, 60))

            # Rounding
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4)

    dpg.bind_theme(global_theme)


def run():
    """Run the main application loop."""
    create_app()
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Start background workers
    workers.start_system_monitor()
    workers.start_progress_poller()

    # Throttling for updates
    last_system_update = 0
    last_progress_update = 0
    SYSTEM_UPDATE_INTERVAL = 2.0  # seconds
    PROGRESS_UPDATE_INTERVAL = 0.5  # seconds

    # Main render loop
    while dpg.is_dearpygui_running():
        now = time.time()

        # Throttled system stats update
        if now - last_system_update >= SYSTEM_UPDATE_INTERVAL:
            workers.update_system_stats()
            system.update_system_panel()
            last_system_update = now

        # Throttled progress update
        if now - last_progress_update >= PROGRESS_UPDATE_INTERVAL:
            workers.poll_progress_file()
            workers.check_pipeline_status()
            pipeline.update_pipeline_panel()
            progress.update_progress_panel()
            last_progress_update = now

        dpg.render_dearpygui_frame()

    # Cleanup
    workers.stop_all()
    dpg.destroy_context()


if __name__ == "__main__":
    run()

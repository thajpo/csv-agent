"""
System monitor panel - memory, CPU, Docker containers.
"""

import dearpygui.dearpygui as dpg

from src.gui.state import state


def create_system_panel(parent: int | str):
    """Create the system monitor panel."""
    with dpg.child_window(parent=parent, width=250, height=150, border=True):
        dpg.add_text("System Monitor", color=(150, 200, 255))
        dpg.add_separator()

        # Memory
        with dpg.group(horizontal=True):
            dpg.add_text("Memory:")
            dpg.add_text("0.0 / 0.0 GB", tag="system_memory")

        dpg.add_progress_bar(
            tag="system_memory_bar",
            default_value=0.0,
            width=-1,
        )

        # CPU
        with dpg.group(horizontal=True):
            dpg.add_text("CPU:")
            dpg.add_text("0%", tag="system_cpu")

        # Docker
        with dpg.group(horizontal=True):
            dpg.add_text("Containers:")
            dpg.add_text("0 / 30", tag="system_docker")


def update_system_panel():
    """Update system panel with current stats."""
    s = state.system

    # Memory
    mem_text = f"{s.memory_used_gb:.1f} / {s.memory_total_gb:.1f} GB"
    dpg.set_value("system_memory", mem_text)
    dpg.set_value("system_memory_bar", s.memory_percent / 100.0)

    # CPU
    dpg.set_value("system_cpu", f"{s.cpu_percent:.0f}%")

    # Docker
    dpg.set_value("system_docker", f"{s.docker_containers} / {s.docker_max}")

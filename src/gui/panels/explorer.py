"""
Data explorer panel - browse datasets, questions, episodes.
"""

import json
from pathlib import Path

import dearpygui.dearpygui as dpg

from src.gui.state import state
from src.datagen.shared.questions_io import load_questions


def create_explorer_panel(parent: int | str):
    """Create the data explorer panel."""
    with dpg.child_window(parent=parent, width=350, height=-1, border=True):
        dpg.add_text("Data Explorer", color=(150, 200, 255))
        dpg.add_separator()

        # Refresh button
        dpg.add_button(label="Refresh", callback=_refresh_data, width=-1)
        dpg.add_spacer(height=5)

        # Tab bar for different data types
        with dpg.tab_bar():
            # Datasets tab
            with dpg.tab(label="Datasets"):
                with dpg.child_window(tag="datasets_list", height=-1, border=False):
                    dpg.add_text("Click Refresh to load", color=(150, 150, 150))

            # Questions tab
            with dpg.tab(label="Questions"):
                with dpg.child_window(tag="questions_list", height=-1, border=False):
                    dpg.add_text("Click Refresh to load", color=(150, 150, 150))

            # Episodes tab
            with dpg.tab(label="Episodes"):
                with dpg.child_window(tag="episodes_list", height=-1, border=False):
                    dpg.add_text("Click Refresh to load", color=(150, 150, 150))


def _refresh_data():
    """Refresh all data listings."""
    _load_datasets()
    _load_questions()
    _load_episodes()


def _load_datasets():
    """Load and display available datasets."""
    # Clear existing
    dpg.delete_item("datasets_list", children_only=True)

    # Find datasets
    datasets = []

    # data/csv/
    csv_dir = state.data_dir / "csv"
    if csv_dir.exists():
        for p in csv_dir.glob("*.csv"):
            datasets.append(("csv", p.stem, p))
        for p in csv_dir.glob("*/data.csv"):
            datasets.append(("csv", p.parent.name, p))

    # data/kaggle/
    kaggle_dir = state.data_dir / "kaggle"
    if kaggle_dir.exists():
        for p in kaggle_dir.glob("*/data.csv"):
            datasets.append(("kaggle", p.parent.name, p))

    if not datasets:
        dpg.add_text("No datasets found", parent="datasets_list", color=(150, 150, 150))
        return

    # Group by source
    with dpg.tree_node(
        label=f"csv ({sum(1 for d in datasets if d[0] == 'csv')})",
        parent="datasets_list",
        default_open=True,
    ):
        for source, name, path in datasets:
            if source == "csv":
                dpg.add_selectable(
                    label=name,
                    callback=lambda s, d, p=path: _on_dataset_select(p),
                )

    with dpg.tree_node(
        label=f"kaggle ({sum(1 for d in datasets if d[0] == 'kaggle')})",
        parent="datasets_list",
        default_open=True,
    ):
        for source, name, path in datasets:
            if source == "kaggle":
                dpg.add_selectable(
                    label=name,
                    callback=lambda s, d, p=path: _on_dataset_select(p),
                )


def _load_questions():
    """Load and display available questions."""
    dpg.delete_item("questions_list", children_only=True)

    questions_dir = state.questions_dir
    if not questions_dir.exists():
        dpg.add_text(
            "No questions directory", parent="questions_list", color=(150, 150, 150)
        )
        return

    question_files = list(questions_dir.glob("*/questions.json"))
    if not question_files:
        dpg.add_text(
            "No questions found", parent="questions_list", color=(150, 150, 150)
        )
        return

    for qf in sorted(question_files):
        dataset_name = qf.parent.name
        try:
            questions = load_questions(str(qf))
            count = len(questions)
        except Exception:
            count = "?"
            questions = []

        with dpg.tree_node(label=f"{dataset_name} ({count})", parent="questions_list"):
            if isinstance(count, int):
                for i, q in enumerate(questions[:20]):  # Limit to first 20
                    q_text = (
                        q.get("question_text") or q.get("question_mechanical") or ""
                    )[:50]
                    difficulty = q.get("difficulty", "?")
                    dpg.add_selectable(
                        label=f"[{difficulty}] {q_text}...",
                        callback=lambda s, d, qq=q: _on_question_select(qq),
                    )
                if count > 20:
                    dpg.add_text(f"... and {count - 20} more", color=(150, 150, 150))


def _load_episodes():
    """Load and display available episodes."""
    dpg.delete_item("episodes_list", children_only=True)

    episodes_dir = state.episodes_dir
    if not episodes_dir.exists():
        dpg.add_text(
            "No episodes directory", parent="episodes_list", color=(150, 150, 150)
        )
        return

    episode_files = list(episodes_dir.glob("*.jsonl"))
    if not episode_files:
        dpg.add_text("No episodes found", parent="episodes_list", color=(150, 150, 150))
        return

    for ef in sorted(episode_files):
        # Count episodes
        try:
            with open(ef) as f:
                count = sum(1 for _ in f)
        except Exception:
            count = "?"

        with dpg.tree_node(label=f"{ef.name} ({count})", parent="episodes_list"):
            if isinstance(count, int) and count > 0:
                # Load first few episodes
                try:
                    with open(ef) as f:
                        for i, line in enumerate(f):
                            if i >= 20:
                                dpg.add_text(
                                    f"... and {count - 20} more", color=(150, 150, 150)
                                )
                                break
                            ep = json.loads(line)
                            q_text = ep.get("question", {}).get("question_text", "")[
                                :40
                            ]
                            verified = ep.get("verified", False)
                            icon = "[pass]" if verified else "[fail]"
                            color = (100, 255, 100) if verified else (255, 100, 100)
                            dpg.add_selectable(
                                label=f"{icon} {q_text}...",
                                callback=lambda s, d, e=ep: _on_episode_select(e),
                            )
                            dpg.configure_item(dpg.last_item(), color=color)
                except Exception:
                    dpg.add_text("Error loading episodes", color=(255, 100, 100))


def _on_dataset_select(path: Path):
    """Handle dataset selection."""
    state.selected_dataset = str(path)
    # Could show dataset preview in trace viewer


def _on_question_select(question: dict):
    """Handle question selection."""
    state.selected_question = question
    # Show in trace viewer
    from src.gui.panels.trace import show_question

    show_question(question)


def _on_episode_select(episode: dict):
    """Handle episode selection."""
    state.selected_episode = episode
    # Show in trace viewer
    from src.gui.panels.trace import show_trace

    show_trace(episode)

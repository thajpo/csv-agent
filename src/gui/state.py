"""
Shared application state for the GUI.
"""

from dataclasses import dataclass, field
from pathlib import Path
from subprocess import Popen


@dataclass
class DatasetProgress:
    """Progress tracking for a single dataset."""

    name: str
    total: int = 0
    done: int = 0
    verified: int = 0
    failed: int = 0

    @property
    def percentage(self) -> float:
        return (self.done / self.total * 100) if self.total > 0 else 0.0


@dataclass
class PipelineProgress:
    """Progress tracking for the running pipeline."""

    stage: str = ""  # "synthetic_generator", "synthetic_episodes", "episode_gen"
    status: str = "idle"  # "idle", "running", "completed", "failed"
    current_dataset: str = ""
    datasets: dict[str, DatasetProgress] = field(default_factory=dict)
    log_lines: list[str] = field(default_factory=list)
    start_time: float | None = None
    elapsed: float = 0.0


@dataclass
class SystemStats:
    """System resource statistics."""

    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    docker_containers: int = 0
    docker_max: int = 30


@dataclass
class AppState:
    """Global application state."""

    # System monitoring
    system: SystemStats = field(default_factory=SystemStats)

    # Pipeline execution
    pipeline: PipelineProgress = field(default_factory=PipelineProgress)
    process: Popen | None = None
    progress_file: Path = field(default_factory=lambda: Path("/tmp/claude/gui_progress.json"))

    # Data paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    csv_sources: list[Path] = field(default_factory=list)
    questions_dir: Path = field(default_factory=lambda: Path("data/questions_synthetic"))
    episodes_dir: Path = field(default_factory=lambda: Path("data/episodes"))

    # Selected items
    selected_dataset: str | None = None
    selected_question: dict | None = None
    selected_episode: dict | None = None

    # UI state
    log_scroll_to_bottom: bool = True


# Global state instance
state = AppState()

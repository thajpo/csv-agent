"""
Background workers for subprocess management and polling.
"""

import json
import subprocess
import time
from pathlib import Path

import psutil

from src.gui.state import state, DatasetProgress, SystemStats


def update_system_stats():
    """Update system resource statistics."""
    mem = psutil.virtual_memory()
    state.system = SystemStats(
        memory_used_gb=mem.used / (1024**3),
        memory_total_gb=mem.total / (1024**3),
        memory_percent=mem.percent,
        cpu_percent=psutil.cpu_percent(interval=None),
        docker_containers=count_docker_containers(),
        docker_max=30,
    )


def count_docker_containers() -> int:
    """Count running csv-agent Docker containers."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", "label=csv_analysis_env"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            lines = [line for line in result.stdout.strip().split("\n") if line]
            return len(lines)
    except Exception:
        pass
    return 0


def poll_progress_file():
    """Poll the progress file and update state."""
    if not state.progress_file.exists():
        return

    try:
        with open(state.progress_file) as f:
            data = json.load(f)

        state.pipeline.stage = data.get("stage", "")
        state.pipeline.status = data.get("status", "idle")
        state.pipeline.current_dataset = data.get("current_dataset", "")

        # Update dataset progress
        for name, info in data.get("datasets", {}).items():
            state.pipeline.datasets[name] = DatasetProgress(
                name=name,
                total=info.get("total", 0),
                done=info.get("done", 0),
                verified=info.get("verified", 0),
                failed=info.get("failed", 0),
            )

        # Append new log lines
        new_lines = data.get("log_lines", [])
        if new_lines:
            state.pipeline.log_lines.extend(new_lines)
            # Keep last 500 lines
            if len(state.pipeline.log_lines) > 500:
                state.pipeline.log_lines = state.pipeline.log_lines[-500:]

    except (json.JSONDecodeError, IOError):
        pass


def start_pipeline(
    stage: str,
    csv_paths: list[Path] | None = None,
    questions_dir: Path | None = None,
    output_path: Path | None = None,
    n_questions: int | None = None,
    n_workers: int = 4,
    parallel: bool = False,
):
    """Start a pipeline stage in a subprocess."""
    if state.process is not None and state.process.poll() is None:
        return False  # Already running

    # Clear previous state
    state.pipeline.datasets.clear()
    state.pipeline.log_lines.clear()
    state.pipeline.status = "running"
    state.pipeline.stage = stage
    state.pipeline.start_time = time.time()

    # Ensure progress file directory exists
    state.progress_file.parent.mkdir(parents=True, exist_ok=True)

    # Remove old progress file
    if state.progress_file.exists():
        state.progress_file.unlink()

    # Build command
    if stage == "synthetic_generator":
        cmd = ["uv", "run", "python", "-m", "src.datagen.synthetic.generator"]
        if csv_paths:
            cmd.extend(["--csv"] + [str(p) for p in csv_paths])
        if n_questions:
            cmd.extend(["--n-questions", str(n_questions)])
        cmd.extend(["--gui-progress", str(state.progress_file)])

    elif stage == "synthetic_episodes":
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "src.datagen.validate_synthetic",
            "--questions-dir",
            str(questions_dir or state.questions_dir),
            "--output",
            str(output_path or state.episodes_dir / "synthetic.jsonl"),
        ]
        if parallel:
            cmd.append("--parallel")
            cmd.extend(["--n-workers", str(n_workers)])
        cmd.extend(["--gui-progress", str(state.progress_file)])

    elif stage == "episode_gen":
        cmd = ["uv", "run", "python", "-m", "src.datagen.episode_gen"]
        if parallel:
            cmd.append("--parallel")
        cmd.extend(["--gui-progress", str(state.progress_file)])

    else:
        return False

    # Start subprocess
    state.process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=Path.cwd(),
    )

    return True


def stop_pipeline():
    """Stop the running pipeline subprocess."""
    if state.process is not None:
        state.process.terminate()
        try:
            state.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            state.process.kill()
        state.process = None
        state.pipeline.status = "stopped"


def check_pipeline_status():
    """Check if the pipeline subprocess is still running."""
    if state.process is not None:
        retcode = state.process.poll()
        if retcode is not None:
            state.pipeline.status = "completed" if retcode == 0 else "failed"
            state.process = None

    if state.pipeline.start_time:
        state.pipeline.elapsed = time.time() - state.pipeline.start_time


def discover_csv_sources() -> list[Path]:
    """Discover available CSV files."""
    sources = []

    # data/csv/
    csv_dir = state.data_dir / "csv"
    if csv_dir.exists():
        sources.extend(csv_dir.glob("*.csv"))
        sources.extend(csv_dir.glob("*/data.csv"))

    # data/kaggle/
    kaggle_dir = state.data_dir / "kaggle"
    if kaggle_dir.exists():
        sources.extend(kaggle_dir.glob("*/data.csv"))

    state.csv_sources = sorted(sources)
    return state.csv_sources


# Background update flags
_system_monitor_running = False
_progress_poller_running = False


def start_system_monitor():
    """Start the system monitor (called from render loop, not a separate thread)."""
    global _system_monitor_running
    _system_monitor_running = True


def start_progress_poller():
    """Start the progress poller (called from render loop, not a separate thread)."""
    global _progress_poller_running
    _progress_poller_running = True


def stop_all():
    """Stop all background workers."""
    global _system_monitor_running, _progress_poller_running
    _system_monitor_running = False
    _progress_poller_running = False
    stop_pipeline()

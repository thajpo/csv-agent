"""
Progress file writer for GUI communication.

Pipeline scripts use this to write progress updates that the GUI polls.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ProgressWriter:
    """Writes progress updates to a JSON file for the GUI to poll."""

    output_path: Path
    stage: str
    status: str = "running"
    current_dataset: str = ""
    datasets: dict = field(default_factory=dict)
    log_lines: list = field(default_factory=list)
    _max_log_lines: int = 100

    def __post_init__(self):
        self.output_path = Path(self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def set_dataset(self, name: str, total: int):
        """Initialize tracking for a dataset."""
        self.datasets[name] = {
            "total": total,
            "done": 0,
            "verified": 0,
            "failed": 0,
        }
        self._write()

    def set_current(self, name: str):
        """Set the currently processing dataset."""
        self.current_dataset = name
        self._write()

    def update_dataset(
        self,
        name: str,
        done: int | None = None,
        verified: int | None = None,
        failed: int | None = None,
    ):
        """Update progress for a dataset."""
        if name not in self.datasets:
            return

        ds = self.datasets[name]
        if done is not None:
            ds["done"] = done
        if verified is not None:
            ds["verified"] = verified
        if failed is not None:
            ds["failed"] = failed
        self._write()

    def increment(self, name: str, verified: bool = True):
        """Increment done count for a dataset."""
        if name not in self.datasets:
            return

        ds = self.datasets[name]
        ds["done"] += 1
        if verified:
            ds["verified"] += 1
        else:
            ds["failed"] += 1
        self._write()

    def log(self, message: str):
        """Add a log line."""
        self.log_lines.append(message)
        if len(self.log_lines) > self._max_log_lines:
            self.log_lines = self.log_lines[-self._max_log_lines:]
        self._write()

    def set_status(self, status: str):
        """Set the overall status (running, completed, failed)."""
        self.status = status
        self._write()

    def complete(self):
        """Mark the pipeline as complete."""
        self.status = "completed"
        self.current_dataset = ""
        self._write()

    def fail(self, error: str | None = None):
        """Mark the pipeline as failed."""
        self.status = "failed"
        if error:
            self.log(f"ERROR: {error}")
        self._write()

    def _write(self):
        """Write current state to file."""
        data = {
            "stage": self.stage,
            "status": self.status,
            "current_dataset": self.current_dataset,
            "datasets": self.datasets,
            "log_lines": self.log_lines[-50:],  # Only send last 50 lines
        }
        try:
            with open(self.output_path, "w") as f:
                json.dump(data, f)
        except IOError:
            pass  # GUI might be reading, ignore write errors


class NoOpProgressWriter:
    """Dummy writer when --gui-progress is not specified."""

    def set_dataset(self, *args, **kwargs):
        pass

    def set_current(self, *args, **kwargs):
        pass

    def update_dataset(self, *args, **kwargs):
        pass

    def increment(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def set_status(self, *args, **kwargs):
        pass

    def complete(self):
        pass

    def fail(self, *args, **kwargs):
        pass

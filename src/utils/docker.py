"""
Docker container utilities.

Shared utilities for managing Docker containers in the CSV analysis pipeline.
"""

import subprocess


def cleanup_csv_sandbox_containers() -> None:
    """
    Clean up all CSV sandbox containers.

    Stops and removes all containers with names matching 'csv-sandbox'.
    The shell command uses 2>/dev/null to handle case when no containers exist.
    """
    subprocess.run(
        "docker stop $(docker ps -q --filter 'name=csv-sandbox') 2>/dev/null && "
        "docker rm $(docker ps -aq --filter 'name=csv-sandbox') 2>/dev/null",
        shell=True,
        capture_output=True
    )

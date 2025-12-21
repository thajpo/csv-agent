"""
Docker container utilities.

Shared utilities for managing Docker containers in the CSV analysis pipeline.
"""

import subprocess


def cleanup_csv_sandbox_containers() -> None:
    """
    Clean up all CSV sandbox containers.
    
    Silently stops and removes all containers with names matching 'csv-sandbox'.
    Safe to call at any time - ignores errors if no containers exist.
    """
    try:
        subprocess.run(
            "docker stop $(docker ps -q --filter 'name=csv-sandbox') 2>/dev/null && "
            "docker rm $(docker ps -aq --filter 'name=csv-sandbox') 2>/dev/null",
            shell=True,
            capture_output=True
        )
    except Exception:
        pass  # Ignore cleanup errors

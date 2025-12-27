"""
Docker container utilities.

Shared utilities for managing Docker containers in the CSV analysis pipeline.
"""

import subprocess


# All container name prefixes used by csv-agent.
# Both LocalCSVAnalysisEnv and MultiTenantContainer must use prefixes from this list.
CONTAINER_PREFIXES = ("csv-sandbox", "csv-mt")


def cleanup_csv_sandbox_containers() -> None:
    """
    Clean up all CSV sandbox containers.

    Stops and removes all containers with names matching any csv-agent prefix.
    """
    for prefix in CONTAINER_PREFIXES:
        subprocess.run(
            f"docker stop $(docker ps -q --filter 'name={prefix}') 2>/dev/null; "
            f"docker rm $(docker ps -aq --filter 'name={prefix}') 2>/dev/null",
            shell=True,
            capture_output=True,
        )

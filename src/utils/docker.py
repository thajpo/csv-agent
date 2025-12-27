"""
Docker container utilities.

Shared utilities for managing Docker containers in the CSV analysis pipeline.

Session-based isolation:
    Each script generates a unique session ID at startup. Container names include
    this session ID, allowing multiple scripts to run concurrently without
    interfering with each other's containers during cleanup.

    Container naming: {prefix}-{session_id}-{uuid}
    Example: csv-mt-epgen-a1b2c3d4-deadbeef
"""

import logging
import subprocess
import uuid

logger = logging.getLogger(__name__)

# All container name prefixes used by csv-agent.
# Both LocalCSVAnalysisEnv and MultiTenantContainer must use prefixes from this list.
CONTAINER_PREFIXES = ("csv-sandbox", "csv-mt")


def generate_session_id() -> str:
    """
    Generate a unique session ID for container isolation.

    Each script should call this once at startup and pass the session_id
    to container creation functions. This allows concurrent scripts to
    clean up only their own containers.

    Returns:
        8-character hex string (e.g., "a1b2c3d4")
    """
    return uuid.uuid4().hex[:8]


def cleanup_session(session_id: str) -> None:
    """
    Clean up containers belonging to a specific session.

    Only stops/removes containers whose names contain the session_id.
    Safe to call while other sessions are running.

    Args:
        session_id: The session ID to clean up (from generate_session_id())
    """
    for prefix in CONTAINER_PREFIXES:
        # Match containers with this session ID in the name
        # Pattern: {prefix}-{session_id}-*
        filter_pattern = f"{prefix}-{session_id}"
        result = subprocess.run(
            f"docker stop $(docker ps -q --filter 'name={filter_pattern}') 2>/dev/null; "
            f"docker rm $(docker ps -aq --filter 'name={filter_pattern}') 2>/dev/null",
            shell=True,
            capture_output=True,
        )
        if result.returncode != 0 and result.stderr:
            stderr = result.stderr.decode().strip()
            if stderr:
                logger.debug(f"Session cleanup for '{filter_pattern}': {stderr}")


def cleanup_csv_sandbox_containers() -> None:
    """
    Clean up ALL CSV sandbox containers (nuclear option).

    Stops and removes all containers with names matching any csv-agent prefix,
    regardless of session. Use cleanup_session() instead for session-scoped cleanup.

    WARNING: This will kill containers from ALL running scripts.
    Only use for manual cleanup or when you're sure no other scripts are running.
    """
    for prefix in CONTAINER_PREFIXES:
        result = subprocess.run(
            f"docker stop $(docker ps -q --filter 'name={prefix}') 2>/dev/null; "
            f"docker rm $(docker ps -aq --filter 'name={prefix}') 2>/dev/null",
            shell=True,
            capture_output=True,
        )
        if result.returncode != 0 and result.stderr:
            stderr = result.stderr.decode().strip()
            if stderr:
                logger.debug(f"Container cleanup for '{prefix}': {stderr}")

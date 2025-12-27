"""
Docker container utilities.

Shared utilities for managing Docker containers in the CSV analysis pipeline.

Session-based isolation:
    Each script generates a unique session ID at startup. Container names include
    this session ID, allowing multiple scripts to run concurrently without
    interfering with each other's containers during cleanup.

    Container naming: {prefix}-{session_id}-{uuid}
    Example: csv-mt-epgen-a1b2c3d4-deadbeef

Resource management:
    Use check_resource_availability() before starting containers to avoid
    OOM conditions when multiple scripts run concurrently.
"""

import logging
import subprocess
import uuid
from dataclasses import dataclass

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


# =============================================================================
# Resource Management
# =============================================================================

# Estimated memory per container (conservative estimate for pandas/numpy workloads)
ESTIMATED_MEMORY_PER_CONTAINER_GB = 1.5

# Minimum free memory to leave for system stability
MIN_FREE_MEMORY_GB = 2.0


@dataclass
class ResourceStatus:
    """Result of resource availability check."""

    existing_containers: int
    available_memory_gb: float
    recommended_max_containers: int
    warning: str | None = None

    @property
    def ok(self) -> bool:
        """True if resources look sufficient."""
        return self.warning is None


def count_csv_agent_containers() -> int:
    """
    Count all running csv-agent containers (across all sessions).

    Returns:
        Number of running containers matching csv-agent prefixes.
    """
    total = 0
    for prefix in CONTAINER_PREFIXES:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"name={prefix}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            total += len(result.stdout.strip().split("\n"))
    return total


def get_available_memory_gb() -> float:
    """
    Get available system memory in GB.

    Returns:
        Available memory in GB, or -1 if unable to determine.
    """
    import platform

    try:
        if platform.system() == "Darwin":  # macOS
            # Use vm_stat for macOS
            result = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, check=True
            )
            # Parse "Pages free" line
            lines = result.stdout.split("\n")
            page_size = 4096  # macOS uses 4KB pages
            free_pages = 0
            inactive_pages = 0

            for line in lines:
                if "Pages free:" in line:
                    free_pages = int(line.split(":")[1].strip().rstrip("."))
                elif "Pages inactive:" in line:
                    inactive_pages = int(line.split(":")[1].strip().rstrip("."))

            # Available = free + inactive (inactive can be reclaimed)
            available_bytes = (free_pages + inactive_pages) * page_size
            return available_bytes / (1024**3)

        else:  # Linux
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        # Value is in kB
                        kb = int(line.split()[1])
                        return kb / (1024**2)
    except Exception as e:
        logger.warning(f"Could not determine available memory: {e}")

    return -1  # Unknown


def check_resource_availability(
    requested_containers: int,
    memory_per_container_gb: float = ESTIMATED_MEMORY_PER_CONTAINER_GB,
) -> ResourceStatus:
    """
    Check if system has resources for the requested number of containers.

    Call this before starting a new batch of containers to avoid OOM.

    Args:
        requested_containers: Number of new containers you want to start.
        memory_per_container_gb: Estimated memory per container (default: 1.5GB).

    Returns:
        ResourceStatus with recommendations and optional warning.

    Example:
        status = check_resource_availability(4)
        if status.warning:
            print(f"⚠️  {status.warning}")
            print(f"Recommended: use --max-concurrent {status.recommended_max_containers}")
    """
    existing = count_csv_agent_containers()
    available_mem = get_available_memory_gb()

    # Calculate how many containers we can safely run
    if available_mem > 0:
        usable_memory = max(0, available_mem - MIN_FREE_MEMORY_GB)
        max_new_containers = int(usable_memory / memory_per_container_gb)
        total_after = existing + requested_containers
        total_memory_needed = total_after * memory_per_container_gb

        warning = None

        if available_mem < MIN_FREE_MEMORY_GB + (requested_containers * memory_per_container_gb):
            warning = (
                f"Low memory: {available_mem:.1f}GB available, "
                f"need ~{requested_containers * memory_per_container_gb:.1f}GB for {requested_containers} containers. "
                f"{existing} containers already running from other sessions."
            )
        elif existing > 0:
            # Not critical, but informational
            logger.info(
                f"Found {existing} existing csv-agent containers from other sessions. "
                f"Available memory: {available_mem:.1f}GB"
            )

        return ResourceStatus(
            existing_containers=existing,
            available_memory_gb=available_mem,
            recommended_max_containers=max(1, max_new_containers),
            warning=warning,
        )
    else:
        # Couldn't determine memory, just report container count
        warning = None
        if existing > 4:
            warning = (
                f"{existing} containers already running from other sessions. "
                f"Consider reducing parallelism to avoid resource exhaustion."
            )

        return ResourceStatus(
            existing_containers=existing,
            available_memory_gb=-1,
            recommended_max_containers=4,  # Conservative default
            warning=warning,
        )

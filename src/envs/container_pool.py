"""
Multi-tenant container for efficient parallel processing.

A single Docker container running multiple worker processes via fork().
Workers share memory (copy-on-write) for loaded packages, significantly
reducing memory usage compared to separate containers.

Architecture:
    MultiTenantContainer (one container per CSV)
        └── Parent process (loads packages + CSV)
            ├── Worker 0 (FIFO pair 0) - for gold trace
            ├── Worker 1 (FIFO pair 1) - for consistency trace 1
            ├── Worker 2 (FIFO pair 2) - for consistency trace 2
            └── ... (n_workers total)

Usage:
    # One container per CSV, with 6 workers for triangulation
    container = MultiTenantContainer(csv_path, n_workers=6)
    await container.start()

    # Run code on workers in parallel
    results = await asyncio.gather(*[
        container.run_on_worker(i, "df.shape") for i in range(6)
    ])

    # Reset workers for next question
    await container.reset_all_workers()

    # Cleanup
    await container.stop()
"""

import asyncio
import base64
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.envs.csv_env import SETUP_CODE, PACKAGES


@dataclass
class Slot:
    """Represents a single worker slot in the pool."""
    container_id: str
    worker_id: int
    in_use: bool = False

    @property
    def slot_id(self) -> str:
        """Unique identifier for this slot."""
        return f"{self.container_id}:{self.worker_id}"

    @property
    def cmd_fifo(self) -> str:
        return f"/tmp/worker_{self.worker_id}_cmd"

    @property
    def res_fifo(self) -> str:
        return f"/tmp/worker_{self.worker_id}_res"


class MultiTenantContainer:
    """
    A single Docker container with multiple worker processes.

    Workers share memory via fork() and copy-on-write semantics,
    significantly reducing memory usage compared to separate containers.

    Use one MultiTenantContainer per CSV file for parallel triangulation.
    """

    IMAGE_NAME = "csv-agent-sandbox"
    DOCKERFILE_PATH = str(Path(__file__).parent / "Dockerfile")

    _build_lock = asyncio.Lock()
    _image_checked = False

    # Multi-tenant worker script - parent forks children after loading packages
    _WORKER_SCRIPT = '''
import ast
import builtins
import contextlib
import io
import json
import os
import signal
import sys
from pathlib import Path
import traceback

N_WORKERS = {n_workers}
READY_FLAG = "/tmp/pool_ready"

# ============================================================================
# SANDBOX SECURITY: Restricted builtins and import whitelist
# ============================================================================

ALLOWED_IMPORTS = frozenset({{
    "pandas", "numpy", "scipy", "sklearn", "statsmodels",
    "json", "math", "re", "collections", "functools", "itertools",
    "datetime", "time", "hashlib", "decimal", "fractions",
    "statistics", "random", "string", "operator", "copy",
}})

_original_import = builtins.__import__

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    top_level = name.split(".")[0]
    if top_level not in ALLOWED_IMPORTS:
        raise ImportError(
            f"Import of '{{name}}' is not allowed. "
            f"Allowed modules: {{', '.join(sorted(ALLOWED_IMPORTS))}}"
        )
    return _original_import(name, globals, locals, fromlist, level)

SAFE_BUILTINS = {{
    "True": True, "False": False, "None": None,
    "Ellipsis": Ellipsis, "NotImplemented": NotImplemented,
    "bool": bool, "int": int, "float": float, "complex": complex,
    "str": str, "bytes": bytes, "bytearray": bytearray,
    "list": list, "tuple": tuple, "set": set, "frozenset": frozenset,
    "dict": dict, "type": type, "object": object,
    "slice": slice, "range": range, "memoryview": memoryview,
    "len": len, "iter": iter, "next": next,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "reversed": reversed, "sorted": sorted,
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "pow": pow, "divmod": divmod,
    "all": all, "any": any,
    "repr": repr, "ascii": ascii, "chr": chr, "ord": ord,
    "format": format, "bin": bin, "hex": hex, "oct": oct,
    "hash": hash, "id": id,
    "hasattr": hasattr, "getattr": getattr, "setattr": setattr, "delattr": delattr,
    "isinstance": isinstance, "issubclass": issubclass,
    "callable": callable, "vars": vars, "dir": dir,
    "print": print, "input": None,
    "Exception": Exception, "BaseException": BaseException,
    "TypeError": TypeError, "ValueError": ValueError,
    "KeyError": KeyError, "IndexError": IndexError,
    "AttributeError": AttributeError, "ImportError": ImportError,
    "RuntimeError": RuntimeError, "StopIteration": StopIteration,
    "ZeroDivisionError": ZeroDivisionError, "AssertionError": AssertionError,
    "NameError": NameError, "LookupError": LookupError,
    "ArithmeticError": ArithmeticError, "OverflowError": OverflowError,
    "FloatingPointError": FloatingPointError,
    "NotImplementedError": NotImplementedError,
    "SyntaxError": SyntaxError, "UnicodeError": UnicodeError,
    "__import__": _safe_import,
}}

def create_restricted_namespace():
    return {{"__name__": "__main__", "__builtins__": SAFE_BUILTINS}}

# ============================================================================
# PARENT PROCESS: Load packages, then fork workers
# ============================================================================

# Heavy imports - done BEFORE fork so children share via CoW
import pandas as pd
import numpy as np
import scipy
from scipy import stats
import sklearn
import statsmodels
import statsmodels.api as sm

# Load CSV - also shared via CoW (read-only)
df = pd.read_csv("/data.csv")
print(f"Parent loaded CSV: {{df.shape[0]}} rows, {{df.shape[1]}} columns", file=sys.stderr)

# Read setup code from file (written by host before fork)
with open("/tmp/setup_code.py", "r") as _f:
    SETUP_INJECT = _f.read()

def run_worker(worker_id: int):
    """Worker process main loop."""
    cmd_fifo = f"/tmp/worker_{{worker_id}}_cmd"
    res_fifo = f"/tmp/worker_{{worker_id}}_res"
    ready_flag = f"/tmp/worker_{{worker_id}}_ready"

    # Create FIFOs
    for path in (cmd_fifo, res_fifo):
        if os.path.exists(path):
            os.remove(path)
        os.mkfifo(path)

    # Initialize namespace with restricted builtins
    namespace = create_restricted_namespace()

    # Inject shared data and setup code
    namespace["df"] = df  # Shared via CoW
    namespace["pd"] = pd
    namespace["np"] = np
    namespace["scipy"] = scipy
    namespace["stats"] = stats
    namespace["sklearn"] = sklearn
    namespace["statsmodels"] = statsmodels
    namespace["sm"] = sm

    # Run setup code (defines submit, hook, normalize_value)
    exec(SETUP_INJECT, namespace, namespace)

    execution_count = 0

    # Signal ready
    Path(ready_flag).write_text("ready")

    while True:
        try:
            with open(cmd_fifo, "r") as f:
                payload = f.read()
            if not payload:
                continue

            request = json.loads(payload)

            if request.get("shutdown"):
                break

            if request.get("reset"):
                # Reset namespace but keep shared data
                namespace = create_restricted_namespace()
                namespace["df"] = df
                namespace["pd"] = pd
                namespace["np"] = np
                namespace["scipy"] = scipy
                namespace["stats"] = stats
                namespace["sklearn"] = sklearn
                namespace["statsmodels"] = statsmodels
                namespace["sm"] = sm
                exec(SETUP_INJECT, namespace, namespace)
                execution_count = 0
                with open(res_fifo, "w") as f:
                    f.write(json.dumps({{"status": "ok", "reset": True}}))
                continue

            code = request.get("code", "")
            execution_count += 1

            result = {{
                "status": "ok",
                "stdout": "",
                "stderr": "",
                "result": None,
                "execution_count": execution_count,
            }}

            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()

            try:
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    module_ast = ast.parse(code, mode="exec")
                    body = list(module_ast.body)
                    trailing_expr = None
                    if body and isinstance(body[-1], ast.Expr):
                        trailing_expr = body.pop()
                    if body:
                        exec(compile(ast.Module(body=body, type_ignores=[]), "<cell>", "exec"), namespace, namespace)
                    if trailing_expr is not None:
                        value = eval(compile(ast.Expression(trailing_expr.value), "<cell>", "eval"), namespace, namespace)
                        if value is not None:
                            result["result"] = repr(value)
            except Exception:
                result["status"] = "error"
                result["result"] = traceback.format_exc()

            result["stdout"] = stdout_buf.getvalue()
            result["stderr"] = stderr_buf.getvalue()

            with open(res_fifo, "w") as f:
                f.write(json.dumps(result))

        except Exception as e:
            print(f"Worker {{worker_id}} error: {{e}}", file=sys.stderr)

# ============================================================================
# MAIN: Fork workers
# ============================================================================

children = []
for i in range(N_WORKERS):
    pid = os.fork()
    if pid == 0:
        # Child process
        run_worker(i)
        sys.exit(0)
    else:
        children.append(pid)

print(f"Parent forked {{N_WORKERS}} workers: {{children}}", file=sys.stderr)

# Wait for all workers to be ready
import time
for i in range(N_WORKERS):
    ready_flag = f"/tmp/worker_{{i}}_ready"
    while not os.path.exists(ready_flag):
        time.sleep(0.1)
    print(f"Worker {{i}} ready", file=sys.stderr)

# Signal overall readiness
Path(READY_FLAG).write_text("ready")
print("All workers ready", file=sys.stderr)

# Parent waits for children (or handles signals)
def handle_signal(signum, frame):
    for pid in children:
        os.kill(pid, signal.SIGTERM)
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

# Wait for children
for pid in children:
    os.waitpid(pid, 0)
'''

    def __init__(
        self,
        csv_path: str,
        n_workers: int = 6,
    ):
        """
        Initialize a multi-tenant container.

        Args:
            csv_path: Path to CSV file to load
            n_workers: Number of worker processes (default: 6 for 1 gold + 5 consistency)
        """
        self.csv_path = Path(csv_path).resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.n_workers = n_workers
        self.container_id: str | None = None
        self.workers: list[Slot] = []
        self._lock = asyncio.Lock()
        self._started = False

    async def _run_docker(self, *args: str, check: bool = True) -> tuple[str, str]:
        """Run a docker command."""
        proc = await asyncio.create_subprocess_exec(
            "docker", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if check and proc.returncode != 0:
            raise RuntimeError(f"Docker command failed: {stderr.decode()}")
        return stdout.decode(), stderr.decode()

    @classmethod
    async def _ensure_image(cls) -> None:
        """Ensure the docker image exists, building it if necessary."""
        async with cls._build_lock:
            if cls._image_checked:
                return

            # Check if image exists
            proc = await asyncio.create_subprocess_exec(
                "docker", "image", "inspect", cls.IMAGE_NAME,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await proc.communicate()

            if proc.returncode != 0:
                print(f"Building docker image '{cls.IMAGE_NAME}'...")
                build_proc = await asyncio.create_subprocess_exec(
                    "docker", "build", "-t", cls.IMAGE_NAME, "-f", cls.DOCKERFILE_PATH, ".",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await build_proc.communicate()
                if build_proc.returncode != 0:
                    raise RuntimeError(f"Failed to build docker image: {stderr.decode()}")
                print(f"✓ Built docker image '{cls.IMAGE_NAME}'")

            cls._image_checked = True

    async def _setup_container(self) -> None:
        """Create and start the container with multiple workers."""
        self.container_id = f"csv-mt-{uuid.uuid4().hex[:8]}"

        # Create container
        await self._run_docker(
            "run", "-d",
            "--name", self.container_id,
            self.IMAGE_NAME,
            "tail", "-f", "/dev/null"
        )

        # Copy CSV
        await self._run_docker(
            "cp", str(self.csv_path), f"{self.container_id}:/data.csv"
        )

        # Write setup code as separate file (avoids escaping issues)
        setup_b64 = base64.b64encode(SETUP_CODE.encode()).decode()
        await self._run_docker(
            "exec", self.container_id,
            "python", "-c",
            f"import base64; open('/tmp/setup_code.py', 'w').write(base64.b64decode('{setup_b64}').decode())"
        )

        # Generate worker script
        worker_script = self._WORKER_SCRIPT.format(n_workers=self.n_workers)

        # Write worker script
        script_b64 = base64.b64encode(worker_script.encode()).decode()
        await self._run_docker(
            "exec", self.container_id,
            "python", "-c",
            f"import base64; open('/tmp/pool_worker.py', 'w').write(base64.b64decode('{script_b64}').decode())"
        )

        # Start worker (runs in background, forks children)
        await self._run_docker(
            "exec", "-d", self.container_id,
            "python", "-u", "/tmp/pool_worker.py"
        )

        # Wait for ready
        ready_flag = "/tmp/pool_ready"
        start_time = asyncio.get_event_loop().time()
        timeout = 60

        while asyncio.get_event_loop().time() - start_time < timeout:
            stdout, _ = await self._run_docker(
                "exec", self.container_id, "cat", ready_flag,
                check=False
            )
            if "ready" in stdout:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError(f"Container {self.container_id} workers did not become ready")

        # Create worker slot objects
        self.workers = [
            Slot(container_id=self.container_id, worker_id=i)
            for i in range(self.n_workers)
        ]

    async def start(self):
        """Start the container and workers."""
        if self._started:
            return

        # Ensure Docker image exists
        await self._ensure_image()

        print(f"Starting multi-tenant container: {self.n_workers} workers for {self.csv_path.name}")

        await self._setup_container()

        self._started = True
        print(f"✓ Container ready: {self.container_id} ({self.n_workers} workers)")

    async def stop(self):
        """Stop and remove the container."""
        if self.container_id:
            await self._run_docker("stop", self.container_id, check=False)
            await self._run_docker("rm", self.container_id, check=False)
        self.container_id = None
        self.workers = []
        self._started = False

    def get_worker(self, worker_id: int) -> Slot:
        """Get worker by index."""
        if worker_id < 0 or worker_id >= len(self.workers):
            raise IndexError(f"Worker {worker_id} not found (have {len(self.workers)} workers)")
        return self.workers[worker_id]

    async def run_on_worker(self, worker_id: int, code: str) -> str:
        """Run code on a specific worker by index."""
        worker = self.get_worker(worker_id)
        return await self._run_code_on_slot(worker, code)

    async def _run_code_on_slot(self, slot: Slot, code: str) -> str:
        """Run code on a specific slot (internal)."""
        payload = json.dumps({"code": code})
        payload_b64 = base64.b64encode(payload.encode()).decode()

        # Send command via FIFO
        send_cmd = f'''
import base64
import json
data = base64.b64decode('{payload_b64}').decode()
with open('{slot.cmd_fifo}', 'w') as f:
    f.write(data)
with open('{slot.res_fifo}', 'r') as f:
    print(f.read())
'''
        stdout, stderr = await self._run_docker(
            "exec", slot.container_id, "python", "-c", send_cmd
        )

        try:
            response = json.loads(stdout.strip())
        except json.JSONDecodeError:
            return f"Error: Failed to parse response: {stdout}"

        # Format output
        parts = []
        if response.get("stdout"):
            parts.append(response["stdout"].rstrip())
        if response.get("stderr"):
            parts.append(response["stderr"].rstrip())
        if response.get("status") == "error":
            parts.append(response.get("result", "Unknown error"))
        elif response.get("result"):
            parts.append(f"Out[{response.get('execution_count', 0)}]: {response['result']}")

        return "\n".join(parts) if parts else ""

    async def reset_worker(self, worker_id: int):
        """Reset a worker's namespace for reuse."""
        worker = self.get_worker(worker_id)
        await self._reset_slot(worker)

    async def reset_all_workers(self):
        """Reset all workers' namespaces for reuse (e.g., between questions)."""
        await asyncio.gather(*[
            self._reset_slot(worker) for worker in self.workers
        ])

    async def _reset_slot(self, slot: Slot):
        """Reset a slot's namespace (internal)."""
        payload = json.dumps({"reset": True})
        payload_b64 = base64.b64encode(payload.encode()).decode()

        send_cmd = f'''
import base64
import json
data = base64.b64decode('{payload_b64}').decode()
with open('{slot.cmd_fifo}', 'w') as f:
    f.write(data)
with open('{slot.res_fifo}', 'r') as f:
    print(f.read())
'''
        await self._run_docker(
            "exec", slot.container_id, "python", "-c", send_cmd
        )

    def get_stats(self) -> dict:
        """Get container statistics."""
        return {
            "container_id": self.container_id,
            "csv_path": str(self.csv_path),
            "n_workers": self.n_workers,
            "started": self._started,
        }


class WorkerAdapter:
    """
    Adapts a MultiTenantContainer worker to the LocalCSVAnalysisEnv interface.

    This allows MultiTenantContainer workers to be used in places that expect
    LocalCSVAnalysisEnv instances (e.g., triangulate_teacher's container_pool).

    Usage:
        container = MultiTenantContainer(csv_path, n_workers=6)
        await container.start()

        # Create adapters for each worker
        container_pool = [
            (WorkerAdapter(container, i), WorkerAdapter.create_state(i))
            for i in range(container.n_workers)
        ]

        # Use with triangulate_teacher
        result = await triangulate_teacher(..., container_pool=container_pool)
    """

    def __init__(self, container: MultiTenantContainer, worker_id: int):
        """
        Create an adapter for a specific worker in the container.

        Args:
            container: The MultiTenantContainer instance
            worker_id: The worker index (0 to n_workers-1)
        """
        self.container = container
        self.worker_id = worker_id

    @staticmethod
    def create_state(worker_id: int) -> dict:
        """Create a state dict compatible with the env interface."""
        return {
            "sandbox_id": f"worker-{worker_id}",
            "python_state": {"ready": True, "execution_count": 0},
        }

    async def setup_state(self, state: dict, **kwargs) -> dict:
        """
        Initialize state for this worker.

        Note: The container is already started, so this just returns the state.
        """
        state["sandbox_id"] = f"worker-{self.worker_id}"
        state["python_state"] = {"ready": True, "execution_count": 0}
        return state

    async def python(
        self,
        code: str,
        sandbox_id: str = None,
        python_state: dict = None,
        **kwargs,
    ) -> str:
        """
        Execute code on this worker.

        Args:
            code: Python code to execute
            sandbox_id: Ignored (worker_id is used instead)
            python_state: Optional state dict for execution count tracking
            **kwargs: Ignored for compatibility

        Returns:
            Formatted output string
        """
        result = await self.container.run_on_worker(self.worker_id, code)

        # Update execution count if state provided
        if python_state:
            python_state["execution_count"] = python_state.get("execution_count", 0) + 1

        return result

    async def reset_state(self, state: dict, **kwargs) -> dict:
        """Reset this worker's namespace for reuse between questions."""
        await self.container.reset_worker(self.worker_id)
        state["python_state"] = {"ready": True, "execution_count": 0}
        return state

    async def destroy_sandbox(self, sandbox_id: str) -> None:
        """
        No-op for adapter - the container manages its own lifecycle.

        The container should be stopped via container.stop() after all work.
        """
        pass


# Backwards compatibility alias
ContainerPool = MultiTenantContainer

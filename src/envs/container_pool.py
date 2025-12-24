"""
Multi-tenant container pool for efficient parallel processing.

This module provides a pool of Docker containers, each running multiple
worker processes via fork(). Workers share memory (copy-on-write) for
loaded packages, significantly reducing memory usage.

Architecture:
    Container Pool Manager
        └── Container 0
            └── Parent (loads packages)
                ├── Worker 0 (FIFO pair 0)
                ├── Worker 1 (FIFO pair 1)
                └── ... (workers_per_container)
        └── Container 1
            └── ...
        └── ... (n_containers)

Usage:
    pool = ContainerPool(csv_path, n_containers=2, workers_per_container=6)
    await pool.start()

    # Acquire slots for parallel work
    slots = await pool.acquire_slots(6)  # Get 6 available slots

    # Run code on slots
    results = await asyncio.gather(*[
        pool.run_code(slot, "df.shape") for slot in slots
    ])

    # Release slots
    await pool.release_slots(slots)

    # Cleanup
    await pool.stop()
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


@dataclass
class Container:
    """Represents a Docker container with multiple workers."""
    container_id: str
    csv_path: str
    n_workers: int
    slots: list[Slot] = field(default_factory=list)
    ready: bool = False


class ContainerPool:
    """
    Pool of multi-tenant Docker containers for parallel code execution.

    Each container runs multiple worker processes that share memory via
    fork() and copy-on-write semantics.
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
        n_containers: int = 2,
        workers_per_container: int = 6,
    ):
        self.csv_path = Path(csv_path).resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.n_containers = n_containers
        self.workers_per_container = workers_per_container
        self.containers: list[Container] = []
        self._lock = asyncio.Lock()
        self._started = False

    @property
    def total_slots(self) -> int:
        return self.n_containers * self.workers_per_container

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

    async def _create_container(self, container_idx: int) -> Container:
        """Create and start a single container with multiple workers."""
        container_id = f"csv-pool-{uuid.uuid4().hex[:8]}"

        # Create container
        await self._run_docker(
            "run", "-d",
            "--name", container_id,
            self.IMAGE_NAME,
            "tail", "-f", "/dev/null"
        )

        # Copy CSV
        await self._run_docker(
            "cp", str(self.csv_path), f"{container_id}:/data.csv"
        )

        # Write setup code as separate file (avoids escaping issues)
        setup_b64 = base64.b64encode(SETUP_CODE.encode()).decode()
        await self._run_docker(
            "exec", container_id,
            "python", "-c",
            f"import base64; open('/tmp/setup_code.py', 'w').write(base64.b64decode('{setup_b64}').decode())"
        )

        # Generate worker script
        worker_script = self._WORKER_SCRIPT.format(n_workers=self.workers_per_container)

        # Write worker script
        script_b64 = base64.b64encode(worker_script.encode()).decode()
        await self._run_docker(
            "exec", container_id,
            "python", "-c",
            f"import base64; open('/tmp/pool_worker.py', 'w').write(base64.b64decode('{script_b64}').decode())"
        )

        # Start worker (runs in background, forks children)
        await self._run_docker(
            "exec", "-d", container_id,
            "python", "-u", "/tmp/pool_worker.py"
        )

        # Wait for ready
        ready_flag = "/tmp/pool_ready"
        start = asyncio.get_event_loop().time()
        timeout = 60

        while asyncio.get_event_loop().time() - start < timeout:
            stdout, _ = await self._run_docker(
                "exec", container_id, "cat", ready_flag,
                check=False
            )
            if "ready" in stdout:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError(f"Container {container_id} workers did not become ready")

        # Create slot objects
        container = Container(
            container_id=container_id,
            csv_path=str(self.csv_path),
            n_workers=self.workers_per_container,
            ready=True
        )
        for i in range(self.workers_per_container):
            container.slots.append(Slot(container_id=container_id, worker_id=i))

        return container

    async def start(self):
        """Start the container pool."""
        if self._started:
            return

        # Ensure Docker image exists
        await self._ensure_image()

        print(f"Starting container pool: {self.n_containers} containers × {self.workers_per_container} workers = {self.total_slots} slots")

        # Create containers in parallel
        tasks = [self._create_container(i) for i in range(self.n_containers)]
        self.containers = await asyncio.gather(*tasks)

        self._started = True
        print(f"Container pool ready: {self.total_slots} slots available")

    async def stop(self):
        """Stop and remove all containers."""
        for container in self.containers:
            await self._run_docker("stop", container.container_id, check=False)
            await self._run_docker("rm", container.container_id, check=False)
        self.containers = []
        self._started = False

    async def acquire_slots(self, n: int) -> list[Slot]:
        """Acquire n available slots. Blocks if not enough available."""
        async with self._lock:
            available = []
            for container in self.containers:
                for slot in container.slots:
                    if not slot.in_use:
                        available.append(slot)
                    if len(available) >= n:
                        break
                if len(available) >= n:
                    break

            if len(available) < n:
                raise RuntimeError(f"Not enough slots: requested {n}, available {len(available)}")

            slots = available[:n]
            for slot in slots:
                slot.in_use = True
            return slots

    async def release_slots(self, slots: list[Slot]):
        """Release slots back to the pool."""
        async with self._lock:
            for slot in slots:
                slot.in_use = False

    async def run_code(self, slot: Slot, code: str) -> str:
        """Run code on a specific slot."""
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

    async def reset_slot(self, slot: Slot):
        """Reset a slot's namespace for reuse."""
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
        """Get pool statistics."""
        total = 0
        in_use = 0
        for container in self.containers:
            for slot in container.slots:
                total += 1
                if slot.in_use:
                    in_use += 1
        return {
            "containers": len(self.containers),
            "workers_per_container": self.workers_per_container,
            "total_slots": total,
            "in_use": in_use,
            "available": total - in_use,
        }

"""
CSV Analysis Environment for verifiers.

Extends verifiers.PythonEnv to provide a pre-configured environment
for data analysis tasks with pandas/numpy/scipy pre-loaded.
"""

import asyncio
import json
from pathlib import Path
from typing import Any
import uuid

import verifiers as vf
from datasets import Dataset
from verifiers.envs.python_env import PythonEnv

PACKAGES = "pandas numpy scipy scikit-learn statsmodels"

# Base imports for container - normalize_value is injected dynamically
_SETUP_IMPORTS = """
import pandas as pd
import numpy as np
import scipy
from scipy import stats
import sklearn
import statsmodels
import statsmodels.api as sm
import json
"""

# submit() and helpers - injected after normalize_value
_SETUP_SUBMIT = '''
def json_default(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)  # Convert numpy bool to Python bool for proper JSON serialization
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

# Default precision for float rounding before hashing.
# Must match csv_spec.hashing.DEFAULT_HASH_PRECISION for consistent hashes.
DEFAULT_HASH_PRECISION = 2

def _round_floats(obj, precision):
    """Recursively round all floats in a structure to consistent precision."""
    if isinstance(obj, float):
        return round(obj, precision)
    if isinstance(obj, dict):
        return {k: _round_floats(v, precision) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        rounded = [_round_floats(item, precision) for item in obj]
        return tuple(rounded) if isinstance(obj, tuple) else rounded
    return obj

# Storage for hooks captured during execution
_captured_hooks = []

MAX_OTHER_VALUE_BYTES = 100_000  # 100KB limit for non-DataFrame complex types

def summarize_df(df):
    """
    Create bounded summary of DataFrame for PRM training.

    Returns a dict with shape, columns, dtypes, sample rows, and numeric stats.
    Size is bounded regardless of DataFrame row count.
    """
    summary = {
        "type": "DataFrame",
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "head": df.head(3).values.tolist(),
    }
    # Add numeric summary for up to 5 numeric columns
    numeric_cols = list(df.select_dtypes(include='number').columns[:5])
    if numeric_cols:
        summary["numeric_summary"] = {}
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                summary["numeric_summary"][col] = {
                    "mean": float(col_data.mean()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                }
    return summary

def summarize_series(s):
    """
    Create bounded summary of Series for PRM training.
    """
    summary = {
        "type": "Series",
        "name": s.name,
        "shape": list(s.shape),
        "dtype": str(s.dtype),
        "head": s.head(5).tolist(),
    }
    if s.dtype.kind in 'iufb':  # int, uint, float, bool
        non_null = s.dropna()
        if len(non_null) > 0:
            summary["stats"] = {
                "mean": float(non_null.mean()),
                "min": float(non_null.min()),
                "max": float(non_null.max()),
            }
    return summary

def hook(value, code_line, name=None, description=None, depends_on=None):
    """
    Capture an intermediate checkpoint for RL verification.

    REQUIRED: Call this after computing each critical intermediate result.
    You must provide the exact code that produced the value.

    Args:
        value: The intermediate value to checkpoint (DataFrame, scalar, etc.)
        code_line: REQUIRED - The exact code that produced this value
        name: Optional variable name (e.g., 'df_filtered')
        description: Optional semantic description
        depends_on: List of hook names this depends on (for DAG ordering)

    Value storage policy (for PRM training):
        - Scalars (int, float, str, bool, None): Always stored in full
        - DataFrame/Series: Bounded summary (shape, dtypes, head, stats)
        - Other complex types (dict, list): Stored if < 100KB, else type+size only

    Example:
        df_filtered = df[df['TR'] == 'control']
        hook(df_filtered, "df_filtered = df[df['TR'] == 'control']", name='df_filtered')

        mean_val = df_filtered['TL'].mean()
        hook(mean_val, "mean_val = df_filtered['TL'].mean()", name='mean_val', depends_on=['df_filtered'])

        submit(mean_val)
    """
    import hashlib

    # Get normalized value for hashing (always hash the full value)
    normalized = normalize_value(value)
    # Round floats before hashing to ensure consistent hashes despite FP precision
    rounded = _round_floats(normalized, DEFAULT_HASH_PRECISION)
    full_json = json.dumps(rounded, sort_keys=True, default=json_default)
    value_hash = hashlib.sha256(full_json.encode()).hexdigest()[:16]

    # Determine stored value based on type
    if isinstance(value, pd.DataFrame):
        stored_value = summarize_df(value)
    elif isinstance(value, pd.Series):
        stored_value = summarize_series(value)
    elif isinstance(normalized, (int, float, str, bool, type(None))):
        # Scalars: store as-is
        stored_value = normalized
    else:
        # Other complex types (lists, dicts, numpy arrays)
        value_size = len(full_json.encode('utf-8'))
        if value_size <= MAX_OTHER_VALUE_BYTES:
            stored_value = normalized
        else:
            # Too large - store metadata only
            stored_value = {"type": type(value).__name__, "size_bytes": value_size}

    hook_data = {
        "__csv_agent_hook__": True,
        "variable_name": name,
        "value_hash": value_hash,
        "value": stored_value,
        "description": description,
        "code_line": code_line,
        "depends_on": depends_on or [],
    }
    _captured_hooks.append(hook_data)

    serialized = json.dumps(hook_data, default=json_default)
    print(f"📍 Hook: {serialized}")
    return value  # Pass through for chaining

def submit(answer, **kwargs):
    """
    Submit your final answer. Only call this once.
    
    Args:
        answer: The answer value (number, string, dict).
        **kwargs: specific keys like 'key_lines' (list of code lines) for evidence.
    """
    normalized = normalize_value(answer)
    # Wrap in specific protocol structure
    submission = {"__csv_agent_answer__": normalized}
    submission.update(kwargs)
    
    # Include captured hooks in submission
    if _captured_hooks:
        submission["hooks"] = _captured_hooks.copy()
    
    serialized = json.dumps(submission, default=json_default)
    print(f"✓ Submitted: {serialized}")
    return normalized
'''


def get_setup_code() -> str:
    """
    Generate setup code with normalize_value injected from shared source.

    This ensures the container's normalize_value stays in sync with
    the host-side version used in teacher.py for answer comparison.
    """
    import inspect
    from csv_spec import normalize_value

    # Get source, remove type hints that might not work in container
    source = inspect.getsource(normalize_value)
    # Remove the 'from typing import Any' dependency by simplifying signature
    source = source.replace(") -> Any:", "):")
    source = source.replace("val: Any", "val")

    return _SETUP_IMPORTS + "\n" + source + "\n" + _SETUP_SUBMIT


# Legacy compatibility - but prefer get_setup_code() for dynamic generation
SETUP_CODE = get_setup_code()


class VerifiersCSVAnalysisEnv(PythonEnv):
    """
    A sandboxed Python environment pre-loaded with a CSV dataset.

    Extends verifiers.PythonEnv to:
    1. Pre-install data science packages (pandas, numpy, scipy, etc.)
    2. Pre-load a CSV file into a `df` variable
    3. Provide standard aliases (pd, np, etc.)
    """

    def __init__(
        self,
        csv_path: str,
        pip_install_packages: str = "pandas numpy scipy scikit-learn statsmodels",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the CSV Analysis Environment.

        Args:
            csv_path: Path to the CSV file to load (will be mounted into container).
            pip_install_packages: Additional packages to install.
            **kwargs: Passed to PythonEnv.
        """
        self.csv_path = Path(csv_path).resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Create dummy dataset to satisfy verifiers' evaluation framework
        # This will be replaced with real question datasets when doing actual evaluation
        dummy_dataset = Dataset.from_dict(
            {"question": ["dummy question"], "answer": ["dummy answer"]}
        )

        super().__init__(
            dataset=dummy_dataset,
            pip_install_packages=pip_install_packages,
            **kwargs,
        )

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        """
        Initialize the environment state with pre-loaded CSV.
        """
        state = await super().setup_state(state, **kwargs)

        await self.python(
            code=SETUP_CODE,
            sandbox_id=state["sandbox_id"],
            sandbox_state=state["sandbox_state"],
            python_state=state["python_state"],
        )

        return state


def load_environment(csv_path: str, **kwargs: Any) -> VerifiersCSVAnalysisEnv:
    """
    Factory function to load the CSV Analysis Environment.
    This is the standard entry point for verifiers.

    Args:
        csv_path: Path to the CSV file.
        **kwargs: Additional arguments for the environment.

    Returns:
        Configured CSVAnalysisEnv instance.
    """
    return VerifiersCSVAnalysisEnv(csv_path=csv_path, **kwargs)


class LocalCSVAnalysisEnv:
    """
    Local Docker-based Python environment with persistent REPL.

    Compatible interface with verifiers' PythonEnv but uses local Docker
    instead of Prime Sandboxes. Uses a persistent Python worker that maintains
    a namespace dict for variable persistence across code executions.
    """

    _COMMAND_FIFO = "/tmp/python_env_cmd"
    _RESPONSE_FIFO = "/tmp/python_env_res"
    _READY_FLAG = "/tmp/python_env_ready"
    _WORKER_PATH = "/tmp/python_worker.py"

    IMAGE_NAME = "csv-analysis-env"
    DOCKERFILE_PATH = "src/envs/Dockerfile"

    _build_lock = asyncio.Lock()
    _image_checked = False

    # Worker script that runs inside the container
    # Maintains a namespace dict and communicates via FIFOs
    # SECURITY: Uses restricted builtins and import whitelist
    _WORKER_SCRIPT = """
import ast
import builtins
import contextlib
import io
import json
import os
from pathlib import Path
import traceback

COMMAND_FIFO = "{command_fifo}"
RESPONSE_FIFO = "{response_fifo}"
READY_FLAG = "{ready_flag}"

# ============================================================================
# SANDBOX SECURITY: Restricted builtins and import whitelist
# ============================================================================

# Modules allowed for import by agent code
ALLOWED_IMPORTS = frozenset({{
    # Data science essentials
    "pandas", "numpy", "scipy", "sklearn", "statsmodels",
    # Standard library safe modules
    "json", "math", "re", "collections", "functools", "itertools",
    "datetime", "time", "hashlib", "decimal", "fractions",
    "statistics", "random", "string", "operator", "copy",
}})

_original_import = builtins.__import__

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    \"\"\"Import hook that only allows whitelisted modules.\"\"\"
    # Get the top-level module name
    top_level = name.split(".")[0]

    # Check whitelist
    if top_level not in ALLOWED_IMPORTS:
        raise ImportError(
            f"Import of '{{name}}' is not allowed. "
            f"Allowed modules: {{', '.join(sorted(ALLOWED_IMPORTS))}}"
        )

    return _original_import(name, globals, locals, fromlist, level)

# Safe builtins - excludes dangerous functions
SAFE_BUILTINS = {{
    # Constants
    "True": True, "False": False, "None": None,
    "Ellipsis": Ellipsis, "NotImplemented": NotImplemented,

    # Types
    "bool": bool, "int": int, "float": float, "complex": complex,
    "str": str, "bytes": bytes, "bytearray": bytearray,
    "list": list, "tuple": tuple, "set": set, "frozenset": frozenset,
    "dict": dict, "type": type, "object": object,
    "slice": slice, "range": range, "memoryview": memoryview,

    # Iteration & sequences
    "len": len, "iter": iter, "next": next,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "reversed": reversed, "sorted": sorted,

    # Math & comparison
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "pow": pow, "divmod": divmod,

    # Logic
    "all": all, "any": any, "not": lambda x: not x,

    # String & repr
    "repr": repr, "ascii": ascii, "chr": chr, "ord": ord,
    "format": format, "bin": bin, "hex": hex, "oct": oct,
    "hash": hash, "id": id,

    # Attribute access
    "hasattr": hasattr, "getattr": getattr, "setattr": setattr, "delattr": delattr,
    "isinstance": isinstance, "issubclass": issubclass,
    "callable": callable, "vars": vars, "dir": dir,

    # I/O (print only - no file access)
    "print": print, "input": None,  # Disable input()

    # Exceptions (needed for try/except)
    "Exception": Exception, "BaseException": BaseException,
    "TypeError": TypeError, "ValueError": ValueError,
    "KeyError": KeyError, "IndexError": IndexError,
    "AttributeError": AttributeError, "ImportError": ImportError,
    "RuntimeError": RuntimeError, "StopIteration": StopIteration,
    "ZeroDivisionError": ZeroDivisionError, "AssertionError": AssertionError,
    "NameError": NameError, "LookupError": LookupError,
    "ArithmeticError": ArithmeticError, "OverflowError": OverflowError,
    "FloatingPointError": FloatingPointError, "KeyboardInterrupt": KeyboardInterrupt,
    "NotImplementedError": NotImplementedError, "IndentationError": IndentationError,
    "SyntaxError": SyntaxError, "SystemExit": SystemExit,
    "UnicodeError": UnicodeError, "UnicodeDecodeError": UnicodeDecodeError,
    "UnicodeEncodeError": UnicodeEncodeError,

    # Import (using our safe version)
    "__import__": _safe_import,

    # EXPLICITLY EXCLUDED (security risks):
    # - open, file operations
    # - exec, eval, compile (code execution)
    # - globals, locals (namespace access)
    # - __build_class__ left out to prevent class definition attacks
}}

def _create_restricted_namespace():
    \"\"\"Create a fresh namespace with restricted builtins.\"\"\"
    return {{
        "__name__": "__main__",
        "__builtins__": SAFE_BUILTINS,
    }}

# ============================================================================
# Worker main loop
# ============================================================================

def ensure_fifo(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
    os.mkfifo(path)

for fifo_path in (COMMAND_FIFO, RESPONSE_FIFO):
    ensure_fifo(fifo_path)

Path(READY_FLAG).write_text("ready", encoding="utf-8")

namespace = _create_restricted_namespace()
execution_count = 0

while True:
    with open(COMMAND_FIFO, "r", encoding="utf-8") as command_file:
        payload = command_file.read()
    if not payload:
        continue
    request = json.loads(payload)
    if request.get("shutdown"):
        break
    if request.get("reset"):
        namespace = _create_restricted_namespace()
        execution_count = 0
        with open(RESPONSE_FIFO, "w", encoding="utf-8") as response_file:
            response_file.write(json.dumps({{"status": "ok", "reset": True}}))
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
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            module_ast = ast.parse(code, mode="exec")
            body = list(module_ast.body)
            trailing_expr = None
            if body and isinstance(body[-1], ast.Expr):
                trailing_expr = body.pop()
            if body:
                exec_module = ast.Module(body=body, type_ignores=[])
                exec(compile(exec_module, "<cell>", "exec"), namespace, namespace)
            if trailing_expr is not None:
                value = eval(
                    compile(ast.Expression(trailing_expr.value), "<cell>", "eval"),
                    namespace,
                    namespace,
                )
                if value is not None:
                    result["result"] = repr(value)
    except Exception:
        result["status"] = "error"
        result["result"] = traceback.format_exc()
    result["stdout"] = stdout_buffer.getvalue()
    result["stderr"] = stderr_buffer.getvalue()
    with open(RESPONSE_FIFO, "w", encoding="utf-8") as response_file:
        response_file.write(json.dumps(result))
"""

    def __init__(
        self,
        csv_path: str,
        pip_install_packages: str = PACKAGES,
        session_id: str | None = None,
    ) -> None:
        """
        Initialize the CSV Analysis Environment.

        Args:
            csv_path: Path to the CSV file to load.
            pip_install_packages: Packages to install in container.
            session_id: Session ID for container isolation (default: random).
                       Container name will be: csv-sandbox-{session_id}-{uuid}
        """
        self.csv_path = Path(csv_path).resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        self.pip_install_packages = pip_install_packages
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.execution_count = 0

    async def _run_docker(
        self, *args: str, check: bool = True, timeout: float | None = None
    ) -> tuple[str, str]:
        """Run a docker command asynchronously with optional timeout."""
        proc = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(
                f"Docker command timed out after {timeout}s: docker {' '.join(args[:3])}..."
            )
        if check and proc.returncode != 0:
            raise RuntimeError(f"Docker command failed: {stderr.decode()}")
        return stdout.decode(), stderr.decode()

    @classmethod
    async def _ensure_image(cls) -> None:
        """Ensure the docker image exists, building it if necessary."""
        # Fast path: skip lock if already checked (double-checked locking)
        if cls._image_checked:
            return

        async with cls._build_lock:
            # Re-check after acquiring lock (another task may have built it)
            if cls._image_checked:
                return

            # Check if image exists
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "image",
                "inspect",
                cls.IMAGE_NAME,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()

            if proc.returncode != 0:
                print(f"Building docker image '{cls.IMAGE_NAME}'...")
                # Build image
                build_proc = await asyncio.create_subprocess_exec(
                    "docker",
                    "build",
                    "-t",
                    cls.IMAGE_NAME,
                    "-f",
                    cls.DOCKERFILE_PATH,
                    ".",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await build_proc.communicate()
                if build_proc.returncode != 0:
                    raise RuntimeError(
                        f"Failed to build docker image: {stderr.decode()}"
                    )
                print(f"✓ Built docker image '{cls.IMAGE_NAME}'")

            cls._image_checked = True

    async def _wait_for_worker_ready(
        self, sandbox_id: str, timeout: float = 30.0
    ) -> None:
        """Wait for the Python worker to signal it's ready."""
        import time

        start = time.time()
        while time.time() - start < timeout:
            stdout, _ = await self._run_docker(
                "exec", sandbox_id, "cat", self._READY_FLAG, check=False
            )
            if "ready" in stdout:
                return
            await asyncio.sleep(0.1)
        raise TimeoutError("Python worker failed to start")

    async def setup_state(self, state: dict, **kwargs) -> dict:
        """
        Initialize the environment: create container, copy CSV, start worker.

        Returns state dict with sandbox_id and python_state.
        """
        await self._ensure_image()

        sandbox_id = f"csv-sandbox-{self.session_id}-{uuid.uuid4().hex[:8]}"

        # Start container
        await self._run_docker(
            "run",
            "-d",
            "--name",
            sandbox_id,
            self.IMAGE_NAME,
            "tail",
            "-f",
            "/dev/null",
        )

        # Copy CSV into container
        await self._run_docker("cp", str(self.csv_path), f"{sandbox_id}:/data.csv")

        # Note: pip install skipped as packages are in the image

        # Write worker script into container
        worker_code = self._WORKER_SCRIPT.format(
            command_fifo=self._COMMAND_FIFO,
            response_fifo=self._RESPONSE_FIFO,
            ready_flag=self._READY_FLAG,
        )
        # Use base64 to safely transfer the script
        import base64

        worker_b64 = base64.b64encode(worker_code.encode()).decode()
        await self._run_docker(
            "exec",
            sandbox_id,
            "python",
            "-c",
            f"import base64; open('{self._WORKER_PATH}', 'w').write(base64.b64decode('{worker_b64}').decode())",
        )

        # Start worker in background
        await self._run_docker(
            "exec", "-d", sandbox_id, "python", "-u", self._WORKER_PATH
        )

        # Wait for worker to be ready
        await self._wait_for_worker_ready(sandbox_id)

        # Store state
        state["sandbox_id"] = sandbox_id
        state["python_state"] = {"ready": True, "execution_count": 0}

        # Run setup code (import libraries, load CSV)
        csv_setup = (
            SETUP_CODE
            + """
try:
    df = pd.read_csv("/data.csv", na_values=['?', 'NA', 'N/A', 'na', 'n/a'], keep_default_na=True)
except UnicodeDecodeError:
    df = pd.read_csv("/data.csv", encoding='latin-1', na_values=['?', 'NA', 'N/A', 'na', 'n/a'], keep_default_na=True)
print(f"Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
"""
        )
        await self.python(
            code=csv_setup,
            sandbox_id=sandbox_id,
            python_state=state["python_state"],
        )

        return state

    async def python(
        self, code: str, sandbox_id: str, python_state: dict = None, **kwargs
    ) -> str:
        """
        Execute code in the persistent Python REPL.

        Args:
            code: Python code to execute
            sandbox_id: Container ID
            python_state: State dict (for execution_count tracking)
            **kwargs: Ignored (for API compatibility)

        Returns:
            Formatted output string (stdout, stderr, result)
        """
        import base64

        # Send code to worker via FIFO
        payload = json.dumps({"code": code})
        payload_b64 = base64.b64encode(payload.encode()).decode()

        send_command = f"""
import base64
import json
data = base64.b64decode('{payload_b64}').decode()
with open('{self._COMMAND_FIFO}', 'w') as f:
    f.write(data)
with open('{self._RESPONSE_FIFO}', 'r') as f:
    print(f.read())
"""
        stdout, stderr = await self._run_docker(
            "exec",
            sandbox_id,
            "python",
            "-c",
            send_command,
            timeout=60.0,  # Prevent indefinite hangs on FIFO deadlock
        )

        # Parse response with detailed error context
        try:
            response = json.loads(stdout.strip())
        except json.JSONDecodeError as e:
            stdout_preview = stdout[:200] if stdout else "(empty)"
            stderr_preview = stderr[:200] if stderr else "(empty)"
            return (
                f"Error: Failed to parse worker response at position {e.pos}.\n"
                f"stdout: {stdout_preview}\n"
                f"stderr: {stderr_preview}"
            )

        # Update execution count
        if python_state:
            python_state["execution_count"] = response.get("execution_count", 0)

        # Format output (same as verifiers)
        return self._format_response(response)

    def _format_response(self, response: dict) -> str:
        """Format worker response into output string."""
        parts = []

        stdout = (response.get("stdout") or "").rstrip()
        if stdout:
            parts.append(stdout)

        stderr = (response.get("stderr") or "").rstrip()
        if stderr:
            parts.append(f"stderr:\n{stderr}")

        status = response.get("status")
        result_text = response.get("result")
        execution_count = response.get("execution_count", 0)

        if status == "error" and result_text:
            parts.append(result_text.rstrip())
        elif status == "ok" and result_text is not None:
            parts.append(f"Out[{execution_count}]: {result_text}")

        if not parts:
            parts.append("(no output)")

        return "\n".join(parts)

    async def destroy_sandbox(self, sandbox_id: str) -> None:
        """Stop and remove the container."""
        await self._run_docker("stop", sandbox_id, check=False)
        await self._run_docker("rm", sandbox_id, check=False)

    async def reset(self, sandbox_id: str, python_state: dict = None) -> None:
        """
        Reset the container for reuse: clear namespace and re-run setup code.

        This is much faster than destroying and recreating the container.
        """
        import base64

        # Send reset command to worker
        payload = json.dumps({"reset": True})
        payload_b64 = base64.b64encode(payload.encode()).decode()

        reset_command = f"""
import base64
import json
data = base64.b64decode('{payload_b64}').decode()
with open('{self._COMMAND_FIFO}', 'w') as f:
    f.write(data)
with open('{self._RESPONSE_FIFO}', 'r') as f:
    print(f.read())
"""
        await self._run_docker("exec", sandbox_id, "python", "-c", reset_command)

        # Reset python_state
        if python_state:
            python_state["execution_count"] = 0

        # Re-run setup code (imports + load CSV)
        csv_setup = (
            SETUP_CODE
            + """
try:
    df = pd.read_csv("/data.csv", na_values=['?', 'NA', 'N/A', 'na', 'n/a'], keep_default_na=True)
except UnicodeDecodeError:
    df = pd.read_csv("/data.csv", encoding='latin-1', na_values=['?', 'NA', 'N/A', 'na', 'n/a'], keep_default_na=True)
print(f"Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
"""
        )
        await self.python(csv_setup, sandbox_id, python_state)

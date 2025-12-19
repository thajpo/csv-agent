"""
CSV Analysis Environment for verifiers.

Extends verifiers.PythonEnv to provide a pre-configured environment
for data analysis tasks with pandas/numpy/scipy pre-loaded.
"""
import asyncio
import json
from pathlib import Path
from typing import Any, Optional
import uuid

import verifiers as vf
from datasets import Dataset
from verifiers.envs.python_env import PythonEnv

PACKAGES = "pandas numpy scipy scikit-learn statsmodels"
SETUP_CODE = f'''
import pandas as pd
import numpy as np
import scipy
from scipy import stats
import sklearn
import statsmodels
import statsmodels.api as sm
import json

def normalize_value(val):
    """
    Standardize answer formats for better comparison.
    Converts DataFrames/Series to Dictionaries or Scalars where appropriate.
    """
    if val is None:
        return None

    # Handle Pandas/Numpy types
    if isinstance(val, pd.DataFrame):
        if val.empty:
            return {{}}

        # 1x1 -> scalar
        if val.shape == (1, 1):
            return val.iloc[0, 0]

        # 1 column -> dict (if index meaningful) or list
        if val.shape[1] == 1:
            series = val.iloc[:, 0]
            if not isinstance(val.index, pd.RangeIndex):
                return series.to_dict()
            return series.tolist()

        # 2 columns -> dict {{col0: col1}}
        if val.shape[1] == 2:
            return dict(zip(val.iloc[:, 0], val.iloc[:, 1]))

        # Default: list of records
        return val.to_dict('records')

    if isinstance(val, pd.Series):
        if val.size == 1:
            return val.iloc[0]
        return val.to_dict()

    if isinstance(val, np.generic):
        return val.item()

    return val

def json_default(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

def submit(answer, **kwargs):
    """
    Submit your final answer. Only call this once.
    
    Args:
        answer: The answer value (number, string, dict).
        **kwargs: specific keys like 'key_lines' (list of code lines) for evidence.
    """
    normalized = normalize_value(answer)
    # Wrap in specific protocol structure
    submission = {{"__csv_agent_answer__": normalized}}
    submission.update(kwargs)
    
    serialized = json.dumps(submission, default=json_default)
    print(f"✓ Submitted: {{serialized}}")
    return normalized
'''

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
        dummy_dataset = Dataset.from_dict({
            "question": ["dummy question"],
            "answer": ["dummy answer"]
        })

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


# Backward-compatible alias
CSVAnalysisEnv = VerifiersCSVAnalysisEnv


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
    _WORKER_SCRIPT = '''
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import traceback

COMMAND_FIFO = "{command_fifo}"
RESPONSE_FIFO = "{response_fifo}"
READY_FLAG = "{ready_flag}"

def ensure_fifo(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
    os.mkfifo(path)

for fifo_path in (COMMAND_FIFO, RESPONSE_FIFO):
    ensure_fifo(fifo_path)

Path(READY_FLAG).write_text("ready", encoding="utf-8")

namespace: dict[str, object] = {{"__name__": "__main__"}}
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
        namespace.clear()
        namespace["__name__"] = "__main__"
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
'''

    def __init__(
        self,
        csv_path: str,
        pip_install_packages: str = PACKAGES,
    ) -> None:
        """
        Initialize the CSV Analysis Environment.
        
        Args:
            csv_path: Path to the CSV file to load.
            pip_install_packages: Packages to install in container.
        """
        self.csv_path = Path(csv_path).resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        self.pip_install_packages = pip_install_packages
        self.execution_count = 0

    async def _run_docker(self, *args: str, check: bool = True) -> tuple[str, str]:
        """Run a docker command asynchronously."""
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
                # Build image
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

    async def _wait_for_worker_ready(self, sandbox_id: str, timeout: float = 30.0) -> None:
        """Wait for the Python worker to signal it's ready."""
        import time
        start = time.time()
        while time.time() - start < timeout:
            stdout, _ = await self._run_docker(
                "exec", sandbox_id, "cat", self._READY_FLAG,
                check=False
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
        
        sandbox_id = f"csv-sandbox-{uuid.uuid4().hex[:8]}"
        
        # Start container
        await self._run_docker(
            "run", "-d",
            "--name", sandbox_id,
            self.IMAGE_NAME,
            "tail", "-f", "/dev/null"
        )
        
        # Copy CSV into container
        await self._run_docker(
            "cp",
            str(self.csv_path),
            f"{sandbox_id}:/data.csv"
        )
        
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
            "exec", sandbox_id,
            "python", "-c",
            f"import base64; open('{self._WORKER_PATH}', 'w').write(base64.b64decode('{worker_b64}').decode())"
        )
        
        # Start worker in background
        await self._run_docker(
            "exec", "-d", sandbox_id,
            "python", "-u", self._WORKER_PATH
        )
        
        # Wait for worker to be ready
        await self._wait_for_worker_ready(sandbox_id)
        
        # Store state
        state["sandbox_id"] = sandbox_id
        state["sandbox_state"] = None  # Not used by local env, but expected by question_gen
        state["python_state"] = {"ready": True, "execution_count": 0}
        
        # Run setup code (import libraries, load CSV)
        csv_setup = SETUP_CODE + f'\ndf = pd.read_csv("/data.csv")\nprint(f"Loaded CSV: {{df.shape[0]}} rows, {{df.shape[1]}} columns")'
        await self.python(
            code=csv_setup,
            sandbox_id=sandbox_id,
            python_state=state["python_state"],
        )
        
        return state

    async def python(self, code: str, sandbox_id: str, python_state: dict = None, **kwargs) -> str:
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
        
        send_command = f'''
import base64
import json
data = base64.b64decode('{payload_b64}').decode()
with open('{self._COMMAND_FIFO}', 'w') as f:
    f.write(data)
with open('{self._RESPONSE_FIFO}', 'r') as f:
    print(f.read())
'''
        stdout, stderr = await self._run_docker(
            "exec", sandbox_id, "python", "-c", send_command
        )
        
        # Parse response
        try:
            response = json.loads(stdout.strip())
        except json.JSONDecodeError:
            return f"Error: Failed to parse worker response: {stdout}"
        
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
        
        reset_command = f'''
import base64
import json
data = base64.b64decode('{payload_b64}').decode()
with open('{self._COMMAND_FIFO}', 'w') as f:
    f.write(data)
with open('{self._RESPONSE_FIFO}', 'r') as f:
    print(f.read())
'''
        await self._run_docker("exec", sandbox_id, "python", "-c", reset_command)
        
        # Reset python_state
        if python_state:
            python_state["execution_count"] = 0
        
        # Re-run setup code (imports + load CSV)
        csv_setup = SETUP_CODE + f'\ndf = pd.read_csv("/data.csv")\nprint(f"Loaded CSV: {{df.shape[0]}} rows, {{df.shape[1]}} columns")'
        await self.python(csv_setup, sandbox_id, python_state)
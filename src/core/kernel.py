"""
Minimal Jupyter Kernel Wrapper

This wraps the jupyter_client library to provide a simple interface:
    kernel = JupyterKernel()
    result = kernel.execute("x = 1 + 1")
    result = kernel.execute("print(x)")  # stdout: "2"

Key concepts:
    1. KernelManager - starts a Python process (the "kernel")
    2. KernelClient - sends code to that process, receives outputs
    3. Messages - the kernel sends back different message types:
       - "stream": stdout/stderr output (from print())
       - "execute_result": the repr of the last expression (like REPL)
       - "error": exception info if code crashed
"""

import atexit
from dataclasses import dataclass, field
from jupyter_client import KernelManager
from queue import Empty
from typing import Optional


@dataclass
class ExecutionResult:
    """Result of executing a code cell."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    result: Optional[str] = None  # repr of last expression, if any
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback: list[str] = field(default_factory=list)
    execution_time_ms: int = 0


class JupyterKernel:
    """
    A stateful Python execution environment.
    
    Usage:
        kernel = JupyterKernel()
        kernel.execute("import pandas as pd")
        kernel.execute("df = pd.DataFrame({'a': [1,2,3]})")
        result = kernel.execute("df.head()")
        print(result.result)  # shows the DataFrame repr
        kernel.shutdown()
    """
    
    def __init__(self, timeout: float = 10.0, workdir: Optional[str] = None, csv_path: Optional[str] = None):
        """
        Start a new Python kernel.

        Args:
            timeout: Max seconds to wait for code execution (default 10s)
            workdir: Working directory for the kernel (default None)
            csv_path: Path to CSV file to load automatically (default None)
        """
        self.timeout = timeout
        self.csv_path = csv_path

        # --- Step 1: Create and start the kernel manager ---
        # This spawns a new Python process that will run our code
        self.km = KernelManager(kernel_name='python3', cwd=workdir)
        self.km.start_kernel()

        # --- Step 2: Get a client to talk to the kernel ---
        # The client sends messages (code) and receives responses
        self.kc = self.km.client()
        self.kc.start_channels()

        # Wait for kernel to be ready (sends a "kernel_info_request")
        self.kc.wait_for_ready(timeout=30)

        # --- Step 3: Setup built-in functions and load CSV ---
        if csv_path:
            self.setup_kernel_builtins(csv_path)
            # Define baseline variables to exclude from artifact snapshots
            # These are pre-loaded by setup_kernel_builtins() and exist in every execution
            self.baseline_vars = {
                'np', 'pd', 'scipy', 'sklearn', 'statsmodels', 'sm',  # Libraries
                'df', 'submit', '__SUBMITTED_ANSWER__'  # Kernel builtins
            }
        else:
            self.baseline_vars = set()

        # Register cleanup on program exit
        atexit.register(self.shutdown)

    def _validate_imports(self, code: str) -> tuple[bool, str]:
        """
        Validate that code contains no import statements.

        All required libraries are pre-imported, so import statements are unnecessary
        and should be blocked to guide the LLM toward cleaner code.

        Returns:
            (True, "") if code is safe to execute
            (False, "error message") if code contains any imports
        """
        import ast

        ALLOWED_LIBS = {'pandas', 'pd', 'numpy', 'np', 'scipy', 'sklearn', 'statsmodels', 'sm'}

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Let execution handle syntax errors (will provide better error messages)
            return (True, "")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # Block ALL imports (even allowed ones - they're pre-imported)
                for alias in node.names:
                    base_module = alias.name.split('.')[0]

                    # Provide different message for allowed vs disallowed libraries
                    if base_module in ALLOWED_LIBS:
                        error_msg = f"""Import statement not needed: '{alias.name}' is already pre-imported.

The following libraries are pre-imported and available for use:
  - pandas (as pd)
  - numpy (as np)
  - scipy (including scipy.stats)
  - sklearn (scikit-learn)
  - statsmodels (including statsmodels.api as sm)

Example: Use 'df.head()' directly instead of 'import pandas as pd; df.head()'"""
                    else:
                        error_msg = f"""Import not allowed: '{alias.name}'.

Available libraries (pre-imported, use directly without import):
  - pandas (as pd)
  - numpy (as np)
  - scipy (including scipy.stats)
  - sklearn (scikit-learn)
  - statsmodels (including statsmodels.api as sm)

Example: Use df.describe() directly instead of 'import pandas as pd'"""
                    return (False, error_msg)

            elif isinstance(node, ast.ImportFrom):
                # Block ALL "from X import Y" statements
                if node.module is None:
                    # Relative imports
                    error_msg = """Relative imports are not allowed.

Available libraries (pre-imported, use directly without import):
  - pandas (as pd)
  - numpy (as np)
  - scipy (including scipy.stats)
  - sklearn (scikit-learn)
  - statsmodels (including statsmodels.api as sm)"""
                    return (False, error_msg)

                base_module = node.module.split('.')[0]

                # Provide different message for allowed vs disallowed libraries
                if base_module in ALLOWED_LIBS:
                    error_msg = f"""Import statement not needed: '{node.module}' is already pre-imported.

The following libraries are pre-imported and available for use:
  - pandas (as pd)
  - numpy (as np)
  - scipy (including scipy.stats)
  - sklearn (scikit-learn)
  - statsmodels (including statsmodels.api as sm)

Example: Use 'sklearn.linear_model.LinearRegression()' directly instead of 'from sklearn.linear_model import LinearRegression'"""
                else:
                    error_msg = f"""Import not allowed: 'from {node.module} import ...'.

Available libraries (pre-imported, use directly without import):
  - pandas (as pd)
  - numpy (as np)
  - scipy (including scipy.stats)
  - sklearn (scikit-learn)
  - statsmodels (including statsmodels.api as sm)

Example: Use df.describe() directly instead of 'import pandas as pd'"""
                return (False, error_msg)

        return (True, "")

    def execute(self, code: str, skip_validation: bool = False) -> ExecutionResult:
        """
        Execute code in the kernel and return the result.

        Args:
            code: Python code to execute
            skip_validation: If True, skip import validation (for internal setup)
        """
        import time
        start = time.perf_counter()

        # Validate imports before execution (unless skipped for internal setup)
        if not skip_validation:
            is_valid, error_msg = self._validate_imports(code)
            if not is_valid:
                return ExecutionResult(
                    success=False,
                    error_type="ImportError",
                    error_message=error_msg,
                    execution_time_ms=0,
                )

        msg_id = self.kc.execute(code)
        outputs = self._collect_outputs(msg_id)
        
        elapsed = int((time.perf_counter() - start) * 1000)
        
        if outputs is None:
            return ExecutionResult(
                success=False,
                error_type="TimeoutError",
                error_message=f"Execution timed out after {self.timeout}s",
                execution_time_ms=elapsed,
            )
        
        return ExecutionResult(
            success=(outputs['error_type'] is None),
            stdout=''.join(outputs['stdout']),
            stderr=''.join(outputs['stderr']),
            result=outputs['result'],
            error_type=outputs['error_type'],
            error_message=outputs['error_message'],
            traceback=outputs['traceback'],
            execution_time_ms=elapsed,
        )
    
    def _collect_outputs(self, msg_id: str) -> Optional[dict]:
        """
        Read messages from kernel until execution completes.
        Returns None on timeout, otherwise dict of collected outputs.
        """
        outputs = {
            'stdout': [],
            'stderr': [],
            'result': None,
            'error_type': None,
            'error_message': None,
            'traceback': [],
        }
        
        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=self.timeout)
            except Empty:
                return None
            
            if msg['parent_header'].get('msg_id') != msg_id:
                continue
            
            msg_type = msg['header']['msg_type']
            content = msg['content']
            
            if msg_type == 'stream':
                outputs[content['name']].append(content['text'])
            elif msg_type == 'execute_result':
                outputs['result'] = content['data'].get('text/plain', '')
            elif msg_type == 'error':
                outputs['error_type'] = content['ename']
                outputs['error_message'] = content['evalue']
                outputs['traceback'] = content['traceback']
            elif msg_type == 'status' and content['execution_state'] == 'idle':
                break
        
        return outputs
    
    def reset(self):
        """
        Reset the kernel to a fresh state (clear all variables).
        Faster than creating a new kernel.
        """
        self.kc.execute("%reset -f")
        # Wait for it to complete
        self.kc.get_shell_msg(timeout=5)

    def setup_kernel_builtins(self, csv_path: str):
        """
        Inject helper functions and load CSV into kernel namespace.

        This sets up:
        - submit() function for submitting final answers
        - Loads the CSV file as 'df'
        - Imports pandas and numpy
        """
        builtin_code = f"""# Import allowed libraries (in dependency order)
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import sklearn
import statsmodels
import statsmodels.api as sm

__SUBMITTED_ANSWER__ = None

def submit(answer):
    '''Submit your final answer.'''
    global __SUBMITTED_ANSWER__
    __SUBMITTED_ANSWER__ = answer
    print(f"✓ Submitted: {{answer}}")
    return answer

# Load dataset
df = pd.read_csv({csv_path!r})
print(f"Dataset loaded: {{df.shape[0]}} rows, {{df.shape[1]}} columns")
"""
        # Skip validation for setup code (it contains necessary imports)
        result = self.execute(builtin_code.strip(), skip_validation=True)
        if not result.success:
            raise RuntimeError(f"Failed to setup kernel builtins: {result.error_message}")

    def get_locals(self) -> dict:
        """
        Get all local variables from the kernel namespace.

        Returns:
            Dict of variable names to their values (DataFrames, scalars, and __SUBMITTED_ANSWER__)
        """
        # Extract only serializable variables (DataFrames, scalars, and submitted answer)
        result = self.execute("""
import pickle
import base64
import pandas as pd

# Get variables we care about: DataFrames, scalars, and __SUBMITTED_ANSWER__
_vars_to_serialize = {}
for _k, _v in list(globals().items()):
    if _k.startswith('_') and _k != '__SUBMITTED_ANSWER__':
        continue
    if isinstance(_v, pd.DataFrame):
        _vars_to_serialize[_k] = _v
    elif isinstance(_v, (int, float, str, bool, type(None))):
        _vars_to_serialize[_k] = _v

# Serialize to base64
_serialized = base64.b64encode(pickle.dumps(_vars_to_serialize)).decode('ascii')
_serialized
        """, skip_validation=True)

        if not result.success or not result.result:
            return {}

        # Decode the result
        import base64
        import pickle
        try:
            serialized = result.result.strip().strip("'\"")
            return pickle.loads(base64.b64decode(serialized))
        except Exception:
            return {}

    def snapshot_artifacts(self) -> dict:
        """
        Capture only USER-CREATED DataFrames and scalars as Artifacts.

        Excludes baseline variables (df, pd, np, submit, etc.) that exist
        in every execution to prevent false positive matches.

        Returns:
            Dict of {variable_name: Artifact}
        """
        from src.core.types import Artifact, hash_artifact
        import pandas as pd

        locals_dict = self.get_locals()
        artifacts = {}

        for name, obj in locals_dict.items():
            # Skip baseline variables (df, pd, np, submit, etc.)
            if hasattr(self, 'baseline_vars') and name in self.baseline_vars:
                continue

            if name.startswith('_'):
                continue  # Skip private vars

            if isinstance(obj, pd.DataFrame):
                artifacts[name] = Artifact(
                    name=name,
                    hash=hash_artifact(obj),
                    type='DataFrame'
                )
            elif isinstance(obj, (int, float, str, bool, type(None))):
                artifacts[name] = Artifact(
                    name=name,
                    hash=hash_artifact(obj),
                    type='scalar'
                )

        return artifacts

    def get_final_answer(self):
        """
        Retrieve value passed to submit().

        Returns:
            The value passed to submit(), or None if not called
        """
        locals_dict = self.get_locals()
        return locals_dict.get('__SUBMITTED_ANSWER__', None)

    def shutdown(self):
        """Stop the kernel process."""
        if getattr(self, '_shutdown', False):
            return  # Already shut down
        self._shutdown = True
        
        try:
            atexit.unregister(self.shutdown)
        except Exception:
            pass
        
        if hasattr(self, 'kc') and self.kc is not None:
            self.kc.stop_channels()
            del self.kc
        if hasattr(self, 'km') and self.km is not None:
            self.km.shutdown_kernel(now=True)
            del self.km
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown()


# -----------------------------------------------------------------------------
# Quick test when run directly
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting kernel...")
    kernel = JupyterKernel(timeout=5)
    
    print("\n--- Test 1: Simple assignment ---")
    r = kernel.execute("x = 42")
    print(f"success={r.success}, stdout='{r.stdout}', result={r.result}")
    
    print("\n--- Test 2: Print statement ---")
    r = kernel.execute("print(f'x = {x}')")
    print(f"success={r.success}, stdout='{r.stdout.strip()}'")
    
    print("\n--- Test 3: Expression result ---")
    r = kernel.execute("x * 2")
    print(f"success={r.success}, result={r.result}")
    
    print("\n--- Test 4: Import and use pandas ---")
    r = kernel.execute("import pandas as pd; df = pd.DataFrame({'a': [1,2,3]})")
    print(f"success={r.success}")
    r = kernel.execute("df")
    print(f"result:\n{r.result}")
    
    print("\n--- Test 5: Error handling ---")
    r = kernel.execute("1/0")
    print(f"success={r.success}, error_type={r.error_type}, error_message={r.error_message}")
    
    print("\n--- Test 6: State persists ---")
    r = kernel.execute("x + 100")  # x was set in test 1
    print(f"success={r.success}, result={r.result}")
    
    print("\nShutting down...")
    kernel.shutdown()
    print("Done!")


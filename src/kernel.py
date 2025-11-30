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
    
    def __init__(self, timeout: float = 10.0, workdir: Optional[str] = None):
        """
        Start a new Python kernel.
        
        Args:
            timeout: Max seconds to wait for code execution (default 10s)
        """
        self.timeout = timeout
        
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
        
        # Register cleanup on program exit
        atexit.register(self.shutdown)
    
    def execute(self, code: str) -> ExecutionResult:
        """Execute code in the kernel and return the result."""
        import time
        start = time.perf_counter()
        
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


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
    
    def __init__(self, timeout: float = 10.0):
        """
        Start a new Python kernel.
        
        Args:
            timeout: Max seconds to wait for code execution (default 10s)
        """
        self.timeout = timeout
        
        # --- Step 1: Create and start the kernel manager ---
        # This spawns a new Python process that will run our code
        self.km = KernelManager(kernel_name='python3')
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
        """
        Execute code in the kernel and return the result.
        
        Args:
            code: Python code to execute
            
        Returns:
            ExecutionResult with stdout, stderr, result, and error info
        """
        import time
        start = time.perf_counter()
        
        # --- Step 3: Send the code to the kernel ---
        # This returns a message ID we use to match responses
        msg_id = self.kc.execute(code)
        
        # Collect outputs
        stdout_parts = []
        stderr_parts = []
        result = None
        error_type = None
        error_message = None
        traceback = []
        
        # --- Step 4: Read messages until execution completes ---
        # The kernel sends multiple messages:
        #   - "stream" for print() output
        #   - "execute_result" for the repr of last expression  
        #   - "error" if an exception occurred
        #   - "status" with state="idle" when done
        while True:
            try:
                # Get next message from the IOPub channel (where outputs go)
                msg = self.kc.get_iopub_msg(timeout=self.timeout)
            except Empty:
                # Timeout - kernel took too long
                elapsed = int((time.perf_counter() - start) * 1000)
                return ExecutionResult(
                    success=False,
                    error_type="TimeoutError",
                    error_message=f"Execution timed out after {self.timeout}s",
                    execution_time_ms=elapsed,
                )
            
            # Only process messages for our execution (not other background stuff)
            if msg['parent_header'].get('msg_id') != msg_id:
                continue
            
            msg_type = msg['header']['msg_type']
            content = msg['content']
            
            # --- Handle different message types ---
            
            if msg_type == 'stream':
                # Output from print() or similar
                # content = {"name": "stdout"|"stderr", "text": "..."}
                if content['name'] == 'stdout':
                    stdout_parts.append(content['text'])
                else:
                    stderr_parts.append(content['text'])
            
            elif msg_type == 'execute_result':
                # The repr of the last expression (like typing "x" in REPL)
                # content = {"data": {"text/plain": "2"}, ...}
                result = content['data'].get('text/plain', '')
            
            elif msg_type == 'error':
                # An exception occurred
                # content = {"ename": "ValueError", "evalue": "...", "traceback": [...]}
                error_type = content['ename']
                error_message = content['evalue']
                traceback = content['traceback']
            
            elif msg_type == 'status':
                # Kernel state changed
                # content = {"execution_state": "busy"|"idle"}
                if content['execution_state'] == 'idle':
                    # Execution finished, we can stop listening
                    break
        
        elapsed = int((time.perf_counter() - start) * 1000)
        
        return ExecutionResult(
            success=(error_type is None),
            stdout=''.join(stdout_parts),
            stderr=''.join(stderr_parts),
            result=result,
            error_type=error_type,
            error_message=error_message,
            traceback=traceback,
            execution_time_ms=elapsed,
        )
    
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
        if hasattr(self, 'kc'):
            self.kc.stop_channels()
        if hasattr(self, 'km'):
            self.km.shutdown_kernel(now=True)
    
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


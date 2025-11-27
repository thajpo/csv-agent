"""
Rich terminal display for kernel interactions.

Usage:
    from src.environment import RichKernel
    
    kernel = RichKernel()
    kernel.run("x = 42")
    kernel.run("print(x)")
    kernel.run("x * 2")
    kernel.run("1/0")  # shows error nicely
    kernel.vars()      # show current variables
"""

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich import box
import json

from .kernel import JupyterKernel, ExecutionResult


console = Console()


class RichKernel:
    """JupyterKernel wrapper with rich terminal output."""
    
    def __init__(self, timeout: float = 10.0):
        self.kernel = JupyterKernel(timeout=timeout)
        self.cell_count = 0
    
    def run(self, code: str, show: bool = True) -> ExecutionResult:
        """
        Execute code and display the result.
        
        Args:
            code: Python code to execute
            show: Whether to print output (default True)
        
        Returns:
            ExecutionResult from the kernel
        """
        self.cell_count += 1
        result = self.kernel.execute(code)
        
        if show:
            self._display(code, result)
        
        return result
    
    def _display(self, code: str, result: ExecutionResult):
        """Pretty-print a cell execution."""
        # Header with cell number and status
        status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
        header = f"[dim]In [{self.cell_count}][/dim] {status} [dim]({result.execution_time_ms}ms)[/dim]"
        console.print(header)
        
        # Code with syntax highlighting
        syntax = Syntax(code, "python", theme="monokai", line_numbers=False)
        console.print(Panel(syntax, box=box.ROUNDED, border_style="dim"))
        
        # Output
        if result.stdout:
            console.print(Text(result.stdout, style="green"))
        
        if result.stderr:
            console.print(Text(result.stderr, style="yellow"))
        
        if result.error_type:
            console.print(f"[red bold]{result.error_type}[/red bold]: [red]{result.error_message}[/red]")
        
        if result.result:
            console.print(f"[cyan]→[/cyan] {result.result}")
        
        console.print()  # blank line between cells
    
    def vars(self) -> dict:
        """
        Display current kernel variables.
        
        Returns:
            Dict of {name: (type, info)}
        """
        # Get variables from kernel
        result = self.kernel.execute("""
import json as _json
import types as _types
_vars = {}
_skip = ('In', 'Out', 'get_ipython', 'exit', 'quit', 'open', 'copyright', 'credits', 'license')
for _name, _val in list(globals().items()):
    if _name.startswith('_') or _name in _skip:
        continue
    if isinstance(_val, (_types.ModuleType, _types.FunctionType, _types.BuiltinFunctionType, type)):
        continue
    _type = type(_val).__name__
    if hasattr(_val, 'shape'):
        _info = str(_val.shape)
    elif hasattr(_val, '__len__') and _type != 'str':
        try:
            _info = f"len={len(_val)}"
        except:
            _info = repr(_val)[:50]
    else:
        _info = repr(_val)[:50]
    _vars[_name] = [_type, _info]
print(_json.dumps(_vars))
del _json, _types, _vars, _name, _val, _type, _info, _skip
""")
        
        if not result.success or not result.stdout.strip():
            console.print("[dim]No variables[/dim]")
            return {}
        
        try:
            vars_dict = json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            console.print("[dim]Could not parse variables[/dim]")
            return {}
        
        # Display as table
        table = Table(title="Variables", box=box.SIMPLE)
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Value/Shape", style="green")
        
        for name, (typ, info) in sorted(vars_dict.items()):
            table.add_row(name, typ, info)
        
        console.print(table)
        return vars_dict
    
    def reset(self):
        """Reset the kernel and cell counter."""
        self.kernel.reset()
        self.cell_count = 0
        console.print("[dim]Kernel reset[/dim]")
    
    def shutdown(self):
        """Shutdown the kernel."""
        self.kernel.shutdown()
        console.print("[dim]Kernel shutdown[/dim]")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown()


# -----------------------------------------------------------------------------
# Quick test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    console.print("[bold]Rich Kernel Demo[/bold]\n")
    
    with RichKernel() as k:
        k.run("x = 42")
        k.run("print(f'x = {x}')")
        k.run("x * 2")
        k.run("import pandas as pd")
        k.run("df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})")
        k.run("df")
        k.vars()
        k.run("1/0")  # error


from src.core.kernel import JupyterKernel

def test_kernel_execute():
    kernel = JupyterKernel()
    result = kernel.execute("print('Hello, World!')")
    assert result.stdout == "Hello, World!\n"

def test_kernel_execute_error():
    kernel = JupyterKernel()
    # Write to stderr explicitly
    result = kernel.execute("import sys; sys.stderr.write('Hello, World!\\n')")
    assert result.stderr == "Hello, World!\n"

def test_kernel_reset():
    kernel = JupyterKernel()
    kernel.execute("print('Hello, World!')")
    kernel.reset()
    result = kernel.execute("print('Hello, World!')")
    assert result.stdout == "Hello, World!\n"
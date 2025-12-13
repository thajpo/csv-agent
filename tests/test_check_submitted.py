from src.core.kernel import JupyterKernel

kernel = JupyterKernel(timeout=30, csv_path="data.csv")
kernel.execute("submit(123)")

result = kernel.execute("print(f'__SUBMITTED_ANSWER__ exists: {\"__SUBMITTED_ANSWER__\" in globals()}')")
print(result.stdout)

result2 = kernel.execute("print(f'Value: {globals().get(\"__SUBMITTED_ANSWER__\", \"NOT FOUND\")}')")
print(result2.stdout)

kernel.shutdown()

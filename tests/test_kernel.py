from src.core.kernel import JupyterKernel

# Test kernel
kernel = JupyterKernel(timeout=30, csv_path="csv/data.csv")

# Execute some code
result1 = kernel.execute("mean_tl = df['TL'].mean()\nprint(mean_tl)")
print(f"Exec 1 success: {result1.success}")
print(f"Exec 1 output: {result1.stdout}")

# Call submit
result2 = kernel.execute("submit(mean_tl)")
print(f"Exec 2 success: {result2.success}")
print(f"Exec 2 output: {result2.stdout}")

# Get final answer
final_answer = kernel.get_final_answer()
print(f"Final answer: {final_answer}")

# Get artifacts
artifacts = kernel.snapshot_artifacts()
print(f"Artifacts: {artifacts}")

kernel.shutdown()

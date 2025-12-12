from src.kernel import JupyterKernel

kernel = JupyterKernel(timeout=30, csv_path="data.csv")
kernel.execute("submit(42)")

locals_dict = kernel.get_locals()
print(f"Locals: {list(locals_dict.keys())}")
print(f"__SUBMITTED_ANSWER__ in locals: {'__SUBMITTED_ANSWER__' in locals_dict}")
if '__SUBMITTED_ANSWER__' in locals_dict:
    print(f"Value: {locals_dict['__SUBMITTED_ANSWER__']}")

final_answer = kernel.get_final_answer()
print(f"Final answer: {final_answer}")

kernel.shutdown()

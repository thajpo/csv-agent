from src.core.kernel import JupyterKernel

# Test kernel
kernel = JupyterKernel(timeout=30, csv_path="csv/data.csv")

# Execute some code
result1 = kernel.execute("mean_tl = df['TL'].mean()\nprint(mean_tl)")
print(f"Exec 1 success: {result1.success}")

# Call submit
result2 = kernel.execute("submit(mean_tl)")
print(f"Exec 2 success: {result2.success}")

# Debug: check if __SUBMITTED_ANSWER__ exists
result3 = kernel.execute("print(f'__SUBMITTED_ANSWER__ = {__SUBMITTED_ANSWER__}')")
print(f"Debug result: success={result3.success}, stdout={result3.stdout}")

# Try to get locals manually
result4 = kernel.execute("""
import pickle
import base64
import pandas as pd

# Get variables we care about
_vars_to_serialize = {}
for _k, _v in globals().items():
    if _k.startswith('_') and _k != '__SUBMITTED_ANSWER__':
        continue
    if isinstance(_v, pd.DataFrame):
        _vars_to_serialize[_k] = _v
    elif isinstance(_v, (int, float, str, bool, type(None))):
        _vars_to_serialize[_k] = _v

print(f"Variables to serialize: {list(_vars_to_serialize.keys())}")

try:
    _serialized = base64.b64encode(pickle.dumps(_vars_to_serialize)).decode('ascii')
    print(f"Serialization successful, length: {len(_serialized)}")
    _serialized
except Exception as e:
    print(f"Serialization failed: {e}")
    None
""")
print(f"Manual test: success={result4.success}")
print(f"Manual test stdout: {result4.stdout}")
print(f"Manual test stderr: {result4.stderr}")
print(f"Manual test result: {result4.result}")

kernel.shutdown()

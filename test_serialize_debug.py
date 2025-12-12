from src.kernel import JupyterKernel

kernel = JupyterKernel(timeout=30, csv_path="data.csv")
kernel.execute("x = 42")
kernel.execute("submit(x)")

# Test the serialization logic directly
result = kernel.execute("""
import pickle
import base64
import pandas as pd

# Get variables we care about
_vars_to_serialize = {}
for _k, _v in globals().items():
    print(f"Checking {_k}: starts_with_={_k.startswith('_')}, is __SUBMITTED_ANSWER__={_k == '__SUBMITTED_ANSWER__'}")
    if _k.startswith('_') and _k != '__SUBMITTED_ANSWER__':
        print(f"  -> Skipping {_k}")
        continue
    if isinstance(_v, pd.DataFrame):
        print(f"  -> Adding {_k} as DataFrame")
        _vars_to_serialize[_k] = _v
    elif isinstance(_v, (int, float, str, bool, type(None))):
        print(f"  -> Adding {_k} as scalar, value={_v}")
        _vars_to_serialize[_k] = _v
    else:
        print(f"  -> Not adding {_k}, type={type(_v)}")

print(f"\\nVariables to serialize: {list(_vars_to_serialize.keys())}")
""", skip_validation=True)

print(result.stdout)

kernel.shutdown()

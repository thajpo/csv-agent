"""
CSV Analysis Environment for verifiers.

Extends verifiers.PythonEnv to provide a pre-configured environment
for data analysis tasks with pandas/numpy/scipy pre-loaded.
"""
from pathlib import Path
from typing import Any, Optional

import verifiers as vf
from verifiers.envs.python_env import PythonEnv


class CSVAnalysisEnv(PythonEnv):
    """
    A sandboxed Python environment pre-loaded with a CSV dataset.
    
    Extends verifiers.PythonEnv to:
    1. Pre-install data science packages (pandas, numpy, scipy, etc.)
    2. Pre-load a CSV file into a `df` variable
    3. Provide standard aliases (pd, np, etc.)
    """
    
    def __init__(
        self,
        csv_path: str,
        pip_install_packages: str = "pandas numpy scipy scikit-learn statsmodels",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the CSV Analysis Environment.
        
        Args:
            csv_path: Path to the CSV file to load (will be mounted into container).
            pip_install_packages: Additional packages to install.
            **kwargs: Passed to PythonEnv.
        """
        self.csv_path = Path(csv_path).resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        super().__init__(
            pip_install_packages=pip_install_packages,
            **kwargs,
        )
    
    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        """
        Initialize the environment state with pre-loaded CSV.
        """
        state = await super().setup_state(state, **kwargs)
        
        # Pre-import libraries and load CSV
        setup_code = f'''
import pandas as pd
import numpy as np
import scipy
from scipy import stats
import sklearn
import statsmodels
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv("{self.csv_path}")
print(f"Loaded CSV: {{df.shape[0]}} rows, {{df.shape[1]}} columns")
'''
        # Execute setup code
        await self.python(
            code=setup_code,
            sandbox_id=state["sandbox_id"],
            sandbox_state=state["sandbox_state"],
            python_state=state["python_state"],
        )
        
        return state


def load_environment(csv_path: str, **kwargs: Any) -> CSVAnalysisEnv:
    """
    Factory function to load the CSV Analysis Environment.
    This is the standard entry point for verifiers.
    
    Args:
        csv_path: Path to the CSV file.
        **kwargs: Additional arguments for the environment.
    
    Returns:
        Configured CSVAnalysisEnv instance.
    """
    return CSVAnalysisEnv(csv_path=csv_path, **kwargs)

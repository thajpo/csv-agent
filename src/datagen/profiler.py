"""
Data Profiler for CSV datasets.
Generates a heavy-context "Fact Bundle" for Tier 1 Question Generation.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import create_logger

# Configure logger
logger = create_logger(__name__)


class DataProfiler:
    """Analyzes a CSV dataset to produce a dense fact bundle."""

    def __init__(self, max_tokens_estimate: int = 5000):
        self.max_tokens_estimate = max_tokens_estimate

    def analyze(self, csv_path: str) -> Dict[str, Any]:
        """
        Analyze a CSV file and return a structured profile.
        
        Args:
            csv_path: Path to the CSV file.
            
        Returns:
            Dictionary containing schema, stats, correlations, and quality metrics.
        """
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load data
        try:
            # First load with default settings
            df = pd.read_csv(csv_file)
            
            # Attempt to coerce object columns to more specific types
            df = self._coerce_types(df)
            
            profile = {
                "dataset_name": csv_file.stem,
                "shape": {"rows": len(df), "columns": len(df.columns)},
                "columns": {},
                "correlations": [],
                "quality_alerts": [],
                "sample_head": df.head(3).to_dict(orient="records")
            }

            # Column Analysis
            for col in df.columns:
                profile["columns"][col] = self._analyze_column(df[col])

            # Multivariate Analysis (Correlations)
            profile["correlations"] = self._compute_correlations(df)

            # Data Quality Checks
            profile["quality_alerts"] = self._check_quality(df)

            return profile

        except Exception as e:
            logger.error(f"Profiling failed for {csv_path}: {e}")
            raise

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to convert object columns to numeric or datetime."""
        for col in df.select_dtypes(include=['object']).columns:
            # Try Numeric
            try:
                # remove common non-numeric chars like '$', ','
                clean_col = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
                numeric_col = pd.to_numeric(clean_col, errors='raise')
                df[col] = numeric_col
                continue
            except (ValueError, TypeError):
                pass

            # Try Datetime
            try:
                datetime_col = pd.to_datetime(df[col], errors='raise')
                df[col] = datetime_col
                continue
            except (ValueError, TypeError):
                pass
                
        return df

    def _analyze_column(self, series: pd.Series) -> Dict[str, Any]:
        """Compute stats for a single column."""
        stats_dict = {
            "dtype": str(series.dtype),
            "missing_count": int(series.isnull().sum()),
            "missing_pct": round(series.isnull().mean() * 100, 2),
            "unique_count": int(series.nunique()),
        }

        # Numeric Stats
        if pd.api.types.is_numeric_dtype(series):
            clean_series = series.dropna()
            if not clean_series.empty:
                stats_dict.update({
                    "mean": float(clean_series.mean()),
                    "std": float(clean_series.std()),
                    "min": float(clean_series.min()),
                    "max": float(clean_series.max()),
                    "q25": float(clean_series.quantile(0.25)),
                    "median": float(clean_series.median()),
                    "q75": float(clean_series.quantile(0.75)),
                    "skew": float(clean_series.skew()),
                    "type": "numeric"
                    # outliers could be added here
                })
        
        # Datetime Stats
        elif pd.api.types.is_datetime64_any_dtype(series):
             clean_series = series.dropna()
             if not clean_series.empty:
                 stats_dict.update({
                     "min": str(clean_series.min()),
                     "max": str(clean_series.max()),
                     "range_days": (clean_series.max() - clean_series.min()).days,
                     "type": "datetime"
                 })

        # Categorical/Text Stats
        else:
            value_counts = series.value_counts().head(5).to_dict()
            stats_dict.update({
                "top_values": {str(k): int(v) for k, v in value_counts.items()},
                "type": "categorical"
            })
            
        return stats_dict

    def _compute_correlations(self, df: pd.DataFrame, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find highest correlations between numeric columns."""
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.shape[1] < 2:
            return []
            
        corr_matrix = numeric_df.corr().abs()
        
        # Select upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Flatten and sort
        pairs = upper.stack().reset_index()
        pairs.columns = ['col1', 'col2', 'correlation']
        pairs = pairs.sort_values('correlation', ascending=False).head(top_k)
        
        return pairs.to_dict(orient='records')

    def _check_quality(self, df: pd.DataFrame) -> List[str]:
        """Identify potential data quality issues."""
        alerts = []
        
        # Duplicate Rows
        dupes = df.duplicated().sum()
        if dupes > 0:
            alerts.append(f"Found {dupes} duplicate rows")
            
        # Empty Columns
        for col in df.columns:
            if df[col].isnull().all():
                alerts.append(f"Column '{col}' is entirely empty")
                
        # High Cardinality Strings (potential unique IDs treated as category)
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            if df[col].nunique() == len(df):
                alerts.append(f"Column '{col}' appears to be a unique identifier (all values unique)")
                
        return alerts

if __name__ == "__main__":
    # Test on data.csv if run directly
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="csv/data.csv")
    args = parser.parse_args()
    
    profiler = DataProfiler()
    try:
        profile = profiler.analyze(args.csv)
        print(json.dumps(profile, indent=2, default=str))
    except Exception as e:
        print(f"Error: {e}")

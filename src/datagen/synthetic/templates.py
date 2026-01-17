"""
Composition templates for deterministic question generation.

Each template defines a code pattern that:
1. Discovers something about the data (forces exploration)
2. Computes a verifiable result
3. Can be verbalized into a natural language question
"""

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class CompositionTemplate:
    """A template for generating a compositional question."""

    name: str
    description: str  # Human-readable description of what this computes
    code_template: str  # Python code with {placeholders}
    output_type: str  # "scalar", "dict", "list"
    output_schema: str  # Exact format description for the answer
    applicable_when: Callable[[dict], bool]  # Profile -> bool
    n_steps: int
    difficulty: str  # EASY, MEDIUM, HARD, VERY_HARD
    category: str = "compositional"
    tags: list[str] = field(default_factory=list)
    param_sets: list[dict[str, Any]] | None = None
    # Alternative code templates for ambiguous interpretations (e.g., variance vs CV)
    # Each alternative produces a different valid answer for the same question
    alternative_code_templates: list[str] | None = None

    def is_applicable(self, profile: dict) -> bool:
        try:
            return self.applicable_when(profile)
        except Exception as e:
            warnings.warn(
                f"Template '{self.name}' applicable_when raised {type(e).__name__}: {e}"
            )
            return False

    def instantiate(self, profile: dict, params: dict[str, Any] | None = None) -> str:
        """Fill in template placeholders with profile data."""
        # Filter columns once to keep template code aligned with the same eligibility rules.
        numeric_cols = _eligible_numeric_cols(profile)

        # Get categorical columns
        categorical_cols = _eligible_categorical_cols(profile)

        # Pick a target column (first numeric with reasonable variance)
        target_col = numeric_cols[0] if numeric_cols else None

        # Fill placeholders
        code = self.code_template
        if target_col:
            code = code.replace("{target_col}", target_col)
        if numeric_cols:
            code = code.replace("{numeric_cols}", str(numeric_cols))
        if categorical_cols:
            code = code.replace("{categorical_cols}", str(categorical_cols))
        if params:
            for key, value in params.items():
                code = code.replace(f"{{{key}}}", repr(value))

        # Inject id-like column exclusions into templates that scan numeric/object columns.
        id_like_cols = _id_like_cols(profile)
        drop_expr = f".drop(columns={id_like_cols}, errors='ignore')"
        code = code.replace("df.select_dtypes('number')", f"df.select_dtypes('number'){drop_expr}")
        code = code.replace(
            "df.select_dtypes(include=['object', 'category'])",
            f"df.select_dtypes(include=['object', 'category']){drop_expr}",
        )

        # Prepend helper function for index-like column detection (used in column filtering)
        helper_code = '''
import re as _re
def _is_index_column(col_name):
    """Check if column is index-like (Unnamed, ID, Person ID, etc.)."""
    lowered = col_name.lower().strip()
    patterns = [
        r'^unnamed:\\s*\\d+$', r'^index$', r'^id$', r'^_id$',
        r'^row.?id$', r'^person.?id$', r'^serial.?no$', r'^sr.?no$'
    ]
    return any(_re.match(p, lowered) for p in patterns)
'''
        code = helper_code + code

        return code

    def instantiate_alternatives(self, profile: dict, params: dict[str, Any] | None = None) -> list[str]:
        """Instantiate all alternative code templates with same transformations as primary."""
        if not self.alternative_code_templates:
            return []

        results = []
        for alt_template in self.alternative_code_templates:
            # Apply same transformations as instantiate()
            numeric_cols = _eligible_numeric_cols(profile)
            categorical_cols = _eligible_categorical_cols(profile)
            target_col = numeric_cols[0] if numeric_cols else None

            code = alt_template
            if target_col:
                code = code.replace("{target_col}", target_col)
            if numeric_cols:
                code = code.replace("{numeric_cols}", str(numeric_cols))
            if categorical_cols:
                code = code.replace("{categorical_cols}", str(categorical_cols))
            if params:
                for key, value in params.items():
                    code = code.replace(f"{{{key}}}", repr(value))

            # Inject id-like column exclusions
            id_like_cols = _id_like_cols(profile)
            drop_expr = f".drop(columns={id_like_cols}, errors='ignore')"
            code = code.replace("df.select_dtypes('number')", f"df.select_dtypes('number'){drop_expr}")
            code = code.replace(
                "df.select_dtypes(include=['object', 'category'])",
                f"df.select_dtypes(include=['object', 'category']){drop_expr}",
            )

            # Prepend helper function
            helper_code = '''
import re as _re
def _is_index_column(col_name):
    """Check if column is index-like (Unnamed, ID, Person ID, etc.)."""
    lowered = col_name.lower().strip()
    patterns = [
        r'^unnamed:\\s*\\d+$', r'^index$', r'^id$', r'^_id$',
        r'^row.?id$', r'^person.?id$', r'^serial.?no$', r'^sr.?no$'
    ]
    return any(_re.match(p, lowered) for p in patterns)
'''
            code = helper_code + code
            results.append(code)

        return results

    def iter_param_sets(self) -> list[dict[str, Any]]:
        """Return all parameter sets for this template."""
        return self.param_sets or [{}]


def _count_numeric_cols(profile: dict) -> int:
    """Count numeric columns in profile."""
    return len(_eligible_numeric_cols(profile))


def _has_categorical_cols(profile: dict) -> bool:
    """Check if dataset has categorical columns."""
    return len(_eligible_categorical_cols(profile)) > 0


def _is_id_like_column(name: str, info: dict, row_count: int) -> bool:
    """Heuristic: exclude identifiers and near-unique columns from analysis targets."""
    if not name:
        return False

    # Check the profiler's is_index_like flag first (most reliable)
    if info.get("is_index_like", False):
        return True

    lowered = name.strip().lower()
    # Expanded patterns: Unnamed: 0, Person ID, Row ID, etc.
    if re.search(r"(^id$|_id$|^id_|uuid|guid|index$|^unnamed:\s*\d+$|person.?id|row.?id|serial.?no|sr.?no)", lowered):
        return True
    unique_count = info.get("unique_count")
    if row_count and isinstance(unique_count, int):
        if unique_count >= max(2, int(0.98 * row_count)):
            return True
    return False


def _id_like_cols(profile: dict) -> list[str]:
    row_count = profile.get("shape", {}).get("rows", 0) or 0
    return [
        col
        for col, info in profile.get("columns", {}).items()
        if _is_id_like_column(col, info, row_count)
    ]


def _eligible_numeric_cols(profile: dict) -> list[str]:
    row_count = profile.get("shape", {}).get("rows", 0) or 0
    cols = []
    for col, info in profile.get("columns", {}).items():
        if info.get("type") != "numeric":
            continue
        # Drop columns that are effectively empty to avoid brittle targets.
        if info.get("missing_pct", 0) >= 95:
            continue
        if info.get("unique_count", 2) <= 1:
            continue
        if _is_id_like_column(col, info, row_count):
            continue
        cols.append(col)
    return cols


def _eligible_categorical_cols(profile: dict) -> list[str]:
    row_count = profile.get("shape", {}).get("rows", 0) or 0
    cols = []
    for col, info in profile.get("columns", {}).items():
        if info.get("type") != "categorical":
            continue
        # Drop columns that are effectively empty to avoid brittle groupings.
        if info.get("missing_pct", 0) >= 95:
            continue
        if info.get("unique_count", 2) <= 1:
            continue
        if _is_id_like_column(col, info, row_count):
            continue
        cols.append(col)
    return cols


def get_eligible_numeric_columns(profile: dict) -> list[str]:
    """Public helper for dataset gating and template selection."""
    return _eligible_numeric_cols(profile)


def get_eligible_categorical_columns(profile: dict) -> list[str]:
    """Public helper for dataset gating and template selection."""
    return _eligible_categorical_cols(profile)


def _numeric_col_names(profile: dict) -> list[str]:
    return _eligible_numeric_cols(profile)


def _categorical_col_names(profile: dict) -> list[str]:
    return _eligible_categorical_cols(profile)


def _get_numeric_stat(profile: dict, col: str, stat: str) -> float | None:
    col_info = profile.get("columns", {}).get(col, {})
    if col_info.get("type") != "numeric":
        return None
    return col_info.get(stat)


def _pick_numeric_col(
    profile: dict,
    strategy: str = "max_variance",
    exclude: set[str] | None = None,
) -> str | None:
    exclude = exclude or set()
    candidates = [c for c in _numeric_col_names(profile) if c not in exclude]
    if not candidates:
        return None

    columns = profile.get("columns", {})

    if strategy == "max_variance":
        scored = [(c, columns[c].get("std", 0) ** 2) for c in candidates]
        return max(scored, key=lambda x: x[1])[0] if scored else None

    elif strategy == "max_cv":
        scored = []
        for c in candidates:
            mean = columns[c].get("mean", 0)
            std = columns[c].get("std", 0)
            if mean != 0:
                scored.append((c, std / abs(mean)))
        return max(scored, key=lambda x: x[1])[0] if scored else None

    elif strategy == "max_skew":
        scored = [(c, abs(columns[c].get("skew", 0))) for c in candidates]
        return max(scored, key=lambda x: x[1])[0] if scored else None

    elif strategy == "min_skew":
        scored = [(c, abs(columns[c].get("skew", 0))) for c in candidates]
        return min(scored, key=lambda x: x[1])[0] if scored else None

    elif strategy == "max_missing":
        scored = [(c, columns[c].get("missing_pct", 0)) for c in candidates]
        return max(scored, key=lambda x: x[1])[0] if scored else None

    elif strategy == "min_missing":
        scored = [(c, columns[c].get("missing_pct", 0)) for c in candidates]
        return min(scored, key=lambda x: x[1])[0] if scored else None

    return candidates[0] if candidates else None


def _pick_categorical_col(
    profile: dict,
    strategy: str = "moderate_cardinality",
    min_unique: int = 2,
    max_unique: int = 30,
) -> str | None:
    columns = profile.get("columns", {})
    n_rows = profile.get("shape", {}).get("rows", 1)

    candidates = []
    for name in _categorical_col_names(profile):
        info = columns.get(name, {})
        n_unique = info.get("unique_count", 0)
        if min_unique <= n_unique <= max_unique and n_unique < n_rows * 0.8:
            candidates.append((name, n_unique))

    if not candidates:
        return None

    if strategy == "moderate_cardinality":
        target = 10
        return min(candidates, key=lambda x: abs(x[1] - target))[0]

    elif strategy == "high_cardinality":
        return max(candidates, key=lambda x: x[1])[0]

    elif strategy == "low_cardinality":
        return min(candidates, key=lambda x: x[1])[0]

    return candidates[0][0] if candidates else None


def _has_missing_numeric(profile: dict) -> bool:
    return any(
        c.get("type") == "numeric" and c.get("missing_pct", 0) > 0
        for c in profile.get("columns", {}).values()
    )


def _has_moderate_cardinality_categorical(profile: dict, min_groups: int = 3, max_groups: int = 10) -> bool:
    """Check if dataset has a categorical column with suitable cardinality for group-based analysis."""
    for col in _eligible_categorical_cols(profile):
        col_info = profile.get("columns", {}).get(col, {})
        n_unique = col_info.get("unique_count", 0)
        if min_groups <= n_unique <= max_groups:
            return True
    return False


# =============================================================================
# SUPERLATIVE TEMPLATES - Find column with property X, compute Y on it
# =============================================================================

MAX_VARIANCE_MEAN = CompositionTemplate(
    name="max_variance_mean",
    description="Find the numeric column with highest variance (excluding identifier-like columns), then compute its mean",
    output_schema="A single number (the mean), rounded to 3 decimal places",
    code_template="""
# Step 1: Find the column with maximum variance
numeric_cols = df.select_dtypes('number')
variances = numeric_cols.var()
max_var_col = variances.idxmax()
hook(max_var_col, "max_var_col = variances.idxmax()", name='max_var_col')
print(f"Column with highest variance: {max_var_col} (variance: {variances[max_var_col]:.4f})")

# Step 2: Compute the mean of that column
result = df[max_var_col].mean()
hook(result, "result = df[max_var_col].mean()", name='result', depends_on=['max_var_col'])
print(f"Mean of {max_var_col}: {result:.4f}")

submit(round(result, 3))
""".strip(),
    output_type="scalar",
    applicable_when=lambda p: _count_numeric_cols(p) >= 2,
    n_steps=3,
    difficulty="MEDIUM",
    alternative_code_templates=[
        # CV-based alternative: "most variable" interpreted as highest coefficient of variation
        """
# Step 1: Find the column with highest coefficient of variation (CV)
numeric_cols = df.select_dtypes('number')
means = numeric_cols.mean()
stds = numeric_cols.std()
# CV = std / |mean|, only for columns with non-zero mean
cv = stds / means.abs()
cv = cv.replace([float('inf'), float('-inf')], float('nan')).dropna()
max_cv_col = cv.idxmax()
hook(max_cv_col, "max_cv_col = cv.idxmax()", name='max_cv_col')
print(f"Column with highest CV: {max_cv_col} (CV: {cv[max_cv_col]:.4f})")

# Step 2: Compute the mean of that column
result = df[max_cv_col].mean()
hook(result, "result = df[max_cv_col].mean()", name='result', depends_on=['max_cv_col'])
print(f"Mean of {max_cv_col}: {result:.4f}")

submit(round(result, 3))
""".strip(),
        # Std-based alternative: "most variable" interpreted as highest standard deviation
        """
# Step 1: Find the column with highest standard deviation
numeric_cols = df.select_dtypes('number')
stds = numeric_cols.std()
max_std_col = stds.idxmax()
hook(max_std_col, "max_std_col = stds.idxmax()", name='max_std_col')
print(f"Column with highest std: {max_std_col} (std: {stds[max_std_col]:.4f})")

# Step 2: Compute the mean of that column
result = df[max_std_col].mean()
hook(result, "result = df[max_std_col].mean()", name='result', depends_on=['max_std_col'])
print(f"Mean of {max_std_col}: {result:.4f}")

submit(round(result, 3))
""".strip(),
        # Exclude-target alternative: exclude the highest-variance column (likely target) first
        """
# Step 1: Find and exclude the likely target column (highest variance)
numeric_cols = df.select_dtypes('number')
variances = numeric_cols.var()
target_col = variances.idxmax()
print(f"Excluding likely target column: {target_col} (variance: {variances[target_col]:.4f})")

# Step 2: Find highest variance among remaining columns
remaining_cols = numeric_cols.drop(columns=[target_col])
remaining_variances = remaining_cols.var()
max_var_col = remaining_variances.idxmax()
hook(max_var_col, "max_var_col among non-target columns", name='max_var_col')
print(f"Column with highest variance (excluding target): {max_var_col} (variance: {remaining_variances[max_var_col]:.4f})")

# Step 3: Compute the mean of that column
result = df[max_var_col].mean()
hook(result, "result = df[max_var_col].mean()", name='result', depends_on=['max_var_col'])
print(f"Mean of {max_var_col}: {result:.4f}")

submit(round(result, 3))
""".strip(),
    ],
)

MIN_MEAN_COLUMN_STD = CompositionTemplate(
    name="min_mean_column_std",
    description="Find the numeric column with lowest mean (excluding identifier-like columns), then compute its standard deviation",
    output_schema="A single number (the standard deviation), rounded to 3 decimal places",
    code_template="""
# Step 1: Find the column with minimum mean
numeric_cols = df.select_dtypes('number')
means = numeric_cols.mean()
min_mean_col = means.idxmin()
hook(min_mean_col, "min_mean_col = means.idxmin()", name='min_mean_col')
print(f"Column with lowest mean: {min_mean_col} (mean: {means[min_mean_col]:.4f})")

# Step 2: Compute the standard deviation of that column
result = df[min_mean_col].std()
hook(result, "result = df[min_mean_col].std()", name='result', depends_on=['min_mean_col'])
print(f"Std of {min_mean_col}: {result:.4f}")

submit(round(result, 3))
""".strip(),
    output_type="scalar",
    applicable_when=lambda p: _count_numeric_cols(p) >= 2,
    n_steps=3,
    difficulty="MEDIUM",
    alternative_code_templates=[
        # Exclude-target alternative: exclude the highest-variance column (likely target) first
        """
# Step 1: Find and exclude the likely target column (highest variance)
numeric_cols = df.select_dtypes('number')
variances = numeric_cols.var()
target_col = variances.idxmax()
print(f"Excluding likely target column: {target_col}")

# Step 2: Find column with minimum mean among remaining columns
remaining_cols = numeric_cols.drop(columns=[target_col])
means = remaining_cols.mean()
min_mean_col = means.idxmin()
hook(min_mean_col, "min_mean_col among non-target columns", name='min_mean_col')
print(f"Column with lowest mean (excluding target): {min_mean_col} (mean: {means[min_mean_col]:.4f})")

# Step 3: Compute the standard deviation of that column
result = df[min_mean_col].std()
hook(result, "result = df[min_mean_col].std()", name='result', depends_on=['min_mean_col'])
print(f"Std of {min_mean_col}: {result:.4f}")

submit(round(result, 3))
""".strip(),
        # Exclude first numeric column alternative (sometimes used as index)
        """
# Step 1: Exclude the first numeric column (may be index-like)
numeric_cols = df.select_dtypes('number')
first_col = numeric_cols.columns[0]
print(f"Excluding first numeric column: {first_col}")
remaining_cols = numeric_cols.drop(columns=[first_col])

# Step 2: Find column with minimum mean among remaining columns
means = remaining_cols.mean()
min_mean_col = means.idxmin()
hook(min_mean_col, "min_mean_col excluding first column", name='min_mean_col')
print(f"Column with lowest mean (excluding first): {min_mean_col} (mean: {means[min_mean_col]:.4f})")

# Step 3: Compute the standard deviation of that column
result = df[min_mean_col].std()
hook(result, "result = df[min_mean_col].std()", name='result', depends_on=['min_mean_col'])
print(f"Std of {min_mean_col}: {result:.4f}")

submit(round(result, 3))
""".strip(),
    ],
)


# =============================================================================
# CROSS-COLUMN TEMPLATES - Discover relationships between columns
# =============================================================================

STRONGEST_CORRELATION = CompositionTemplate(
    name="strongest_correlation",
    description="Find the pair of numeric columns with the strongest absolute correlation (excluding identifier-like columns)",
    output_schema='A JSON object with exactly two keys: "columns" (a list of the two column names, alphabetically sorted) and "association_score" (the association strength rounded to 2 decimals places). Example: {"columns": ["col_a", "col_b"], "association_score": 0.84}',
    code_template="""
# Step 1: Compute correlation matrix
numeric_cols = df.select_dtypes('number')
corr_matrix = numeric_cols.corr().abs()
hook(corr_matrix.shape, "correlation matrix computed", name='corr_shape')
print(f"Correlation matrix: {corr_matrix.shape[0]} x {corr_matrix.shape[1]}")

# Step 2: Find the maximum off-diagonal correlation
np.fill_diagonal(corr_matrix.values, 0)  # Zero out diagonal
max_corr_idx = corr_matrix.stack().idxmax()
hook(max_corr_idx, "max_corr_idx = corr_matrix.stack().idxmax()", name='max_corr_idx')
print(f"Strongest correlation pair: {max_corr_idx}")

# Step 3: Get the correlation value
correlation_value = corr_matrix.loc[max_corr_idx[0], max_corr_idx[1]]
hook(correlation_value, "correlation_value extracted", name='correlation_value', depends_on=['max_corr_idx'])
print(f"Correlation: {correlation_value:.4f}")

submit({"columns": sorted(list(max_corr_idx)), "association_score": round(float(correlation_value), 2)})
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 3,
    n_steps=4,
    difficulty="MEDIUM",
    alternative_code_templates=[
        # Exclude-target alternative: exclude the highest-variance column (likely target) from correlation
        """
# Step 1: Identify and exclude likely target column (highest variance)
numeric_cols = df.select_dtypes('number')
variances = numeric_cols.var()
target_col = variances.idxmax()
print(f"Excluding likely target column: {target_col}")
feature_cols = numeric_cols.drop(columns=[target_col])

# Step 2: Compute correlation matrix on remaining columns
corr_matrix = feature_cols.corr().abs()
hook(corr_matrix.shape, "correlation matrix computed (excluding target)", name='corr_shape')
print(f"Correlation matrix: {corr_matrix.shape[0]} x {corr_matrix.shape[1]}")

# Step 3: Find the maximum off-diagonal correlation
np.fill_diagonal(corr_matrix.values, 0)
max_corr_idx = corr_matrix.stack().idxmax()
hook(max_corr_idx, "max_corr_idx among features", name='max_corr_idx')
print(f"Strongest correlation pair (features only): {max_corr_idx}")

# Step 4: Get the correlation value
correlation_value = corr_matrix.loc[max_corr_idx[0], max_corr_idx[1]]
hook(correlation_value, "correlation_value extracted", name='correlation_value', depends_on=['max_corr_idx'])
print(f"Correlation: {correlation_value:.4f}")

submit({"columns": sorted(list(max_corr_idx)), "association_score": round(float(correlation_value), 2)})
""".strip(),
        # Use signed correlation (not absolute) - strongest could be most negative
        """
# Step 1: Compute correlation matrix (signed, not absolute)
numeric_cols = df.select_dtypes('number')
corr_matrix = numeric_cols.corr()
hook(corr_matrix.shape, "correlation matrix computed (signed)", name='corr_shape')
print(f"Correlation matrix: {corr_matrix.shape[0]} x {corr_matrix.shape[1]}")

# Step 2: Find the maximum off-diagonal correlation by absolute value
np.fill_diagonal(corr_matrix.values, 0)
abs_corr = corr_matrix.abs()
max_corr_idx = abs_corr.stack().idxmax()
hook(max_corr_idx, "max_corr_idx by absolute value", name='max_corr_idx')
print(f"Strongest correlation pair: {max_corr_idx}")

# Step 3: Get the actual (signed) correlation value, report absolute
correlation_value = abs(corr_matrix.loc[max_corr_idx[0], max_corr_idx[1]])
hook(correlation_value, "correlation_value (absolute)", name='correlation_value', depends_on=['max_corr_idx'])
print(f"Correlation (absolute): {correlation_value:.4f}")

submit({"columns": sorted(list(max_corr_idx)), "association_score": round(float(correlation_value), 2)})
""".strip(),
    ],
)

WEAKEST_CORRELATION = CompositionTemplate(
    name="weakest_correlation",
    description="Find the pair of numeric columns with the weakest absolute correlation (closest to zero) (excluding identifier-like columns)",
    output_schema='A JSON object with exactly two keys: "columns" (a list of the two column names, alphabetically sorted) and "association_score" (the absolute association strength rounded to 2 decimals places). Example: {"columns": ["col_a", "col_b"], "association_score": 0.01}',
    code_template="""
# Step 1: Compute correlation matrix
numeric_cols = df.select_dtypes('number')
corr_matrix = numeric_cols.corr().abs()
hook(corr_matrix.shape, "correlation matrix computed", name='corr_shape')

# Step 2: Find minimum correlation (mask diagonal)
np.fill_diagonal(corr_matrix.values, np.nan)
min_corr_idx = corr_matrix.stack().idxmin()
hook(min_corr_idx, "min_corr_idx found", name='min_corr_idx')
print(f"Weakest correlation pair: {min_corr_idx}")

# Step 3: Get the correlation value
correlation_value = corr_matrix.loc[min_corr_idx[0], min_corr_idx[1]]
hook(correlation_value, "correlation_value extracted", name='correlation_value', depends_on=['min_corr_idx'])
print(f"Correlation: {correlation_value:.4f}")

submit({"columns": sorted(list(min_corr_idx)), "association_score": round(float(correlation_value), 2)})
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 3,
    n_steps=4,
    difficulty="MEDIUM",
    alternative_code_templates=[
        # Exclude-target alternative: exclude the highest-variance column (likely target) from correlation
        """
# Step 1: Identify and exclude likely target column (highest variance)
numeric_cols = df.select_dtypes('number')
variances = numeric_cols.var()
target_col = variances.idxmax()
print(f"Excluding likely target column: {target_col}")
feature_cols = numeric_cols.drop(columns=[target_col])

# Step 2: Compute correlation matrix on remaining columns
corr_matrix = feature_cols.corr().abs()
hook(corr_matrix.shape, "correlation matrix computed (excluding target)", name='corr_shape')

# Step 3: Find minimum correlation (mask diagonal)
np.fill_diagonal(corr_matrix.values, np.nan)
min_corr_idx = corr_matrix.stack().idxmin()
hook(min_corr_idx, "min_corr_idx among features", name='min_corr_idx')
print(f"Weakest correlation pair (features only): {min_corr_idx}")

# Step 4: Get the correlation value
correlation_value = corr_matrix.loc[min_corr_idx[0], min_corr_idx[1]]
hook(correlation_value, "correlation_value extracted", name='correlation_value', depends_on=['min_corr_idx'])
print(f"Correlation: {correlation_value:.4f}")

submit({"columns": sorted(list(min_corr_idx)), "association_score": round(float(correlation_value), 2)})
""".strip(),
        # Use signed correlation (not absolute)
        """
# Step 1: Compute correlation matrix (signed, not absolute)
numeric_cols = df.select_dtypes('number')
corr_matrix = numeric_cols.corr()
hook(corr_matrix.shape, "correlation matrix computed (signed)", name='corr_shape')

# Step 2: Find minimum absolute correlation (mask diagonal)
np.fill_diagonal(corr_matrix.values, np.nan)
abs_corr = corr_matrix.abs()
min_corr_idx = abs_corr.stack().idxmin()
hook(min_corr_idx, "min_corr_idx by absolute value", name='min_corr_idx')
print(f"Weakest correlation pair: {min_corr_idx}")

# Step 3: Get the absolute correlation value
correlation_value = abs(corr_matrix.loc[min_corr_idx[0], min_corr_idx[1]])
hook(correlation_value, "correlation_value (absolute)", name='correlation_value', depends_on=['min_corr_idx'])
print(f"Correlation (absolute): {correlation_value:.4f}")

submit({"columns": sorted(list(min_corr_idx)), "association_score": round(float(correlation_value), 2)})
""".strip(),
    ],
)


# =============================================================================
# CONDITIONAL TEMPLATES - Test condition, branch based on result
# =============================================================================

CONDITIONAL_NORMALITY = CompositionTemplate(
    name="conditional_normality",
    description="Test if highest-variance column (excluding identifier-like columns) is normally distributed using Shapiro-Wilk at alpha=0.05 (sampled to 5000 rows if larger); report mean±std if normal, median+IQR if not",
    output_schema='A JSON object with exactly 3 keys. If normal: {"distribution": "normal", "mean": <number>, "std": <number>}. If non-normal: {"distribution": "non-normal", "median": <number>, "iqr": <number>}. The "iqr" value is Q3 minus Q1 as a single number. All numeric values rounded to 3 decimal places.',
    code_template="""
# Step 1: Get the target column (highest variance numeric column)
numeric_cols = df.select_dtypes('number')
variances = numeric_cols.var()
target_col = variances.idxmax()
hook(target_col, "target_col = variances.idxmax()", name='target_col')
print(f"Testing normality of: {target_col}")

# Step 2: Test for normality (Shapiro-Wilk, sample if large)
sample_data = df[target_col].dropna()
if len(sample_data) > 5000:
    sample_data = sample_data.sample(5000, random_state=42)
_, p_value = scipy.stats.shapiro(sample_data)
hook(p_value, "p_value from Shapiro-Wilk test", name='p_value', depends_on=['target_col'])
print(f"Shapiro-Wilk p-value: {p_value:.6f}")

# Step 3: Branch based on normality
is_normal = p_value > 0.05
hook(is_normal, "is_normal = p_value > 0.05", name='is_normal', depends_on=['p_value'])

if is_normal:
    result = {
        "distribution": "normal",
        "mean": round(df[target_col].mean(), 3),
        "std": round(df[target_col].std(), 3)
    }
else:
    q75, q25 = df[target_col].quantile(0.75), df[target_col].quantile(0.25)
    result = {
        "distribution": "non-normal",
        "median": round(df[target_col].median(), 3),
        "iqr": round(q75 - q25, 3)
    }

hook(result, "conditional result computed", name='result', depends_on=['is_normal'])
print(f"Result: {result}")
submit(result)
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 1,
    n_steps=5,
    difficulty="HARD",
)


# =============================================================================
# THRESHOLD TEMPLATES - Count/filter based on discovered thresholds
# =============================================================================

COUNT_HIGH_MISSING_COLUMNS = CompositionTemplate(
    name="count_high_missing_columns",
    description="Count how many columns have more than 5% missing values",
    output_schema='A JSON object with exactly 2 keys: "count" (integer) and "columns" (list of column names, alphabetically sorted). Example: {"count": 3, "columns": ["col_a", "col_b", "col_c"]}',
    code_template="""
# Step 1: Calculate missing percentage for each column
missing_pct = df.isnull().mean() * 100
hook(missing_pct.to_dict(), "missing percentages computed", name='missing_pct')
print("Missing percentages:")
print(missing_pct.sort_values(ascending=False).head(10))

# Step 2: Count columns above threshold
threshold = {missing_threshold}
high_missing_cols = (missing_pct > threshold).sum()
hook(high_missing_cols, f"count of columns with >{threshold}% missing", name='high_missing_count', depends_on=['missing_pct'])
print(f"Columns with >{threshold}% missing: {high_missing_cols}")

# Step 3: List which columns
high_missing_names = missing_pct[missing_pct > threshold].index.tolist()
hook(high_missing_names, "names of high-missing columns", name='high_missing_names', depends_on=['missing_pct'])

submit({"count": int(high_missing_cols), "columns": sorted(high_missing_names)})
""".strip(),
    output_type="dict",
    applicable_when=lambda p: True,  # Always applicable
    n_steps=3,
    difficulty="EASY",
    param_sets=[
        {"missing_threshold": 5.0},
        {"missing_threshold": 10.0},
    ],
)

DUPLICATE_ROWS_SUMMARY = CompositionTemplate(
    name="duplicate_rows_summary",
    description="Summarize duplicate rows to highlight data cleanliness issues",
    output_schema='A JSON object with exactly 2 keys: "duplicate_rows" (integer) and "duplicate_pct" (percentage rounded to 2 decimals). Example: {"duplicate_rows": 12, "duplicate_pct": 0.48}',
    code_template="""
# Step 1: Count duplicate rows
duplicate_rows = int(df.duplicated().sum())
hook(duplicate_rows, "duplicate_rows counted", name='duplicate_rows')

# Step 2: Compute duplicate percentage
total_rows = len(df)
duplicate_pct = round((duplicate_rows / total_rows) * 100, 2) if total_rows else 0.0
hook(duplicate_pct, "duplicate_pct computed", name='duplicate_pct', depends_on=['duplicate_rows'])

submit({"duplicate_rows": duplicate_rows, "duplicate_pct": duplicate_pct})
""".strip(),
    output_type="dict",
    applicable_when=lambda p: (p.get("shape", {}).get("rows", 0) or 0) >= 2,
    n_steps=3,
    difficulty="EASY",
)

IDENTIFIER_LIKE_COLUMNS = CompositionTemplate(
    name="identifier_like_columns",
    description="Identify columns that behave like row identifiers (near-unique values)",
    output_schema='A JSON object with exactly 2 keys: "count" (integer) and "columns" (list of column names, alphabetically sorted). Example: {"count": 2, "columns": ["id", "user_id"]}',
    code_template="""
# Step 1: Compute uniqueness ratio per column
row_count = len(df)
unique_ratio = df.nunique(dropna=False) / row_count if row_count else pd.Series(dtype=float)
hook(unique_ratio.to_dict(), "unique_ratio computed", name='unique_ratio')

# Step 2: Flag near-unique columns
threshold = {unique_ratio_threshold}
id_like_cols = unique_ratio[unique_ratio >= threshold].index.tolist()
hook(id_like_cols, "id_like_cols identified", name='id_like_cols', depends_on=['unique_ratio'])

submit({"count": len(id_like_cols), "columns": sorted(id_like_cols)})
""".strip(),
    output_type="dict",
    applicable_when=lambda p: (p.get("shape", {}).get("rows", 0) or 0) >= 50,
    n_steps=3,
    difficulty="EASY",
    param_sets=[
        {"unique_ratio_threshold": 0.98},
    ],
)

COUNT_OUTLIER_COLUMNS = CompositionTemplate(
    name="count_outlier_columns",
    description="Count how many numeric columns (excluding identifier-like columns) contain outliers (beyond 3 std from mean)",
    output_schema='A JSON object with exactly 2 keys: "columns_with_outliers" (integer count of columns containing at least one outlier) and "total_outliers" (integer total count of outlier values across all columns). Example: {"columns_with_outliers": 3, "total_outliers": 47}',
    code_template="""
# Step 1: For each numeric column, check for outliers
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
outlier_counts = {}

for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()
    if std > 0:
        outliers = ((df[col] - mean).abs() > {z_threshold} * std).sum()
        outlier_counts[col] = int(outliers)

hook(outlier_counts, "outlier counts per column", name='outlier_counts')
print("Outlier counts per column:")
for col, count in sorted(outlier_counts.items(), key=lambda x: -x[1])[:5]:
    print(f"  {col}: {count}")

# Step 2: Count columns with any outliers
cols_with_outliers = sum(1 for v in outlier_counts.values() if v > 0)
hook(cols_with_outliers, "count of columns with outliers", name='cols_with_outliers', depends_on=['outlier_counts'])
print(f"Columns with outliers: {cols_with_outliers}/{len(numeric_cols)}")

# Step 3: Total outlier count
total_outliers = sum(outlier_counts.values())
hook(total_outliers, "total outliers across all columns", name='total_outliers', depends_on=['outlier_counts'])

submit({"columns_with_outliers": cols_with_outliers, "total_outliers": total_outliers})
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 1,
    n_steps=4,
    difficulty="MEDIUM",
    param_sets=[
        {"z_threshold": 3.0},
        {"z_threshold": 2.5},
    ],
    alternative_code_templates=[
        # IQR-based outliers (1.5 * IQR beyond Q1/Q3) - common alternative definition
        """
# Step 1: For each numeric column, check for outliers using IQR method
numeric_cols = df.select_dtypes('number').columns
outlier_counts = {}

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    outlier_counts[col] = int(outliers)

hook(outlier_counts, "outlier counts per column (IQR method)", name='outlier_counts')
print("Outlier counts per column (IQR method):")
for col, count in sorted(outlier_counts.items(), key=lambda x: -x[1])[:5]:
    print(f"  {col}: {count}")

# Step 2: Count columns with any outliers
cols_with_outliers = sum(1 for v in outlier_counts.values() if v > 0)
hook(cols_with_outliers, "count of columns with outliers", name='cols_with_outliers', depends_on=['outlier_counts'])
print(f"Columns with outliers: {cols_with_outliers}/{len(numeric_cols)}")

# Step 3: Total outlier count
total_outliers = sum(outlier_counts.values())
hook(total_outliers, "total outliers across all columns", name='total_outliers', depends_on=['outlier_counts'])

submit({"columns_with_outliers": cols_with_outliers, "total_outliers": total_outliers})
""".strip(),
        # Inclusive threshold (>= instead of >) for z-score method
        """
# Step 1: For each numeric column, check for outliers (inclusive threshold)
numeric_cols = df.select_dtypes('number').columns
outlier_counts = {}
z_threshold = 3.0

for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()
    if std > 0:
        # Using >= for inclusive threshold interpretation
        outliers = ((df[col] - mean).abs() >= z_threshold * std).sum()
        outlier_counts[col] = int(outliers)

hook(outlier_counts, "outlier counts per column (inclusive)", name='outlier_counts')
print("Outlier counts per column (>= threshold):")
for col, count in sorted(outlier_counts.items(), key=lambda x: -x[1])[:5]:
    print(f"  {col}: {count}")

# Step 2: Count columns with any outliers
cols_with_outliers = sum(1 for v in outlier_counts.values() if v > 0)
hook(cols_with_outliers, "count of columns with outliers", name='cols_with_outliers', depends_on=['outlier_counts'])
print(f"Columns with outliers: {cols_with_outliers}/{len(numeric_cols)}")

# Step 3: Total outlier count
total_outliers = sum(outlier_counts.values())
hook(total_outliers, "total outliers across all columns", name='total_outliers', depends_on=['outlier_counts'])

submit({"columns_with_outliers": cols_with_outliers, "total_outliers": total_outliers})
""".strip(),
        # Using ddof=0 (population std) instead of ddof=1 (sample std)
        """
# Step 1: For each numeric column, check for outliers using population std
numeric_cols = df.select_dtypes('number').columns
outlier_counts = {}
z_threshold = 3.0

for col in numeric_cols:
    mean = df[col].mean()
    # Use population std (ddof=0) instead of sample std (ddof=1)
    std = df[col].std(ddof=0)
    if std > 0:
        outliers = ((df[col] - mean).abs() > z_threshold * std).sum()
        outlier_counts[col] = int(outliers)

hook(outlier_counts, "outlier counts per column (population std)", name='outlier_counts')
print("Outlier counts per column (population std):")
for col, count in sorted(outlier_counts.items(), key=lambda x: -x[1])[:5]:
    print(f"  {col}: {count}")

# Step 2: Count columns with any outliers
cols_with_outliers = sum(1 for v in outlier_counts.values() if v > 0)
hook(cols_with_outliers, "count of columns with outliers", name='cols_with_outliers', depends_on=['outlier_counts'])
print(f"Columns with outliers: {cols_with_outliers}/{len(numeric_cols)}")

# Step 3: Total outlier count
total_outliers = sum(outlier_counts.values())
hook(total_outliers, "total outliers across all columns", name='total_outliers', depends_on=['outlier_counts'])

submit({"columns_with_outliers": cols_with_outliers, "total_outliers": total_outliers})
""".strip(),
    ],
)


# =============================================================================
# MULTI-STAGE TEMPLATES - Discovery + Transform + Aggregate chains
# =============================================================================

CATEGORY_WITH_HIGHEST_TARGET_MEAN = CompositionTemplate(
    name="category_highest_target_mean",
    description="Find which category (from first categorical column with 2-20 unique values) has the highest mean for the highest-variance numeric column (excluding identifier-like columns)",
    output_schema='A JSON object with exactly 4 keys: "category_column" (the grouping column name), "best_category" (the category value with highest mean), "target_column" (the numeric column analyzed), and "mean_value" (rounded to 3 decimal places). Example: {"category_column": "region", "best_category": "West", "target_column": "sales", "mean_value": 1234.567}',
    code_template="""
# Step 1: Identify the target (most variable numeric column)
numeric_cols = df.select_dtypes('number')
variances = numeric_cols.var()
target_col = variances.idxmax()
hook(target_col, "target_col identified", name='target_col')
print(f"Target column (highest variance): {target_col}")

# Step 2: Identify categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
if not categorical_cols:
    submit({"error": "No categorical columns found"})
else:
    # Use the categorical column with reasonable cardinality
    cat_col = None
    for c in categorical_cols:
        if 2 <= df[c].nunique() <= 20:
            cat_col = c
            break

    if cat_col is None:
        cat_col = categorical_cols[0]

    hook(cat_col, "categorical column selected", name='cat_col')
    print(f"Categorical column: {cat_col}")

    # Step 3: Compute mean of target by category
    means_by_category = df.groupby(cat_col)[target_col].mean()
    hook(means_by_category.to_dict(), "means by category", name='means_by_category', depends_on=['target_col', 'cat_col'])
    print(f"Mean {target_col} by {cat_col}:")
    print(means_by_category.sort_values(ascending=False))

    # Step 4: Find the category with highest mean
    best_category = means_by_category.idxmax()
    best_mean = means_by_category.max()
    hook(best_category, "best category identified", name='best_category', depends_on=['means_by_category'])

    submit({
        "category_column": cat_col,
        "best_category": str(best_category),
        "target_column": target_col,
        "mean_value": round(float(best_mean), 3)
    })
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 1 and _has_categorical_cols(p),
    n_steps=5,
    difficulty="HARD",
)

CORRELATION_AFTER_OUTLIER_REMOVAL = CompositionTemplate(
    name="correlation_after_outlier_removal",
    description="Find strongest absolute correlation pair (excluding identifier-like columns), remove outliers beyond 3 standard deviations from both columns, recompute correlation",
    output_schema='A JSON object with exactly 4 keys: "columns" (list of 2 column names, alphabetically sorted), "original_association" (rounded to 2 decimals), "outliers_removed" (integer count), and "clean_association" (rounded to 2 decimals). Example: {"columns": ["col_a", "col_b"], "original_association": 0.8470, "outliers_removed": 12, "clean_association": 0.8910}',
    code_template="""
# Step 1: Find the strongest correlation pair
numeric_cols = df.select_dtypes('number')
corr_matrix = numeric_cols.corr().abs()
np.fill_diagonal(corr_matrix.values, 0)
max_idx = corr_matrix.stack().idxmax()
col1, col2 = max_idx
original_corr = corr_matrix.loc[col1, col2]
hook({"col1": col1, "col2": col2, "original_corr": float(original_corr)}, "strongest pair found", name='original_pair')
print(f"Strongest correlation: {col1} vs {col2} = {original_corr:.4f}")

# Step 2: Remove outliers (beyond 3 std) from both columns
mask1 = (df[col1] - df[col1].mean()).abs() <= 3 * df[col1].std()
mask2 = (df[col2] - df[col2].mean()).abs() <= 3 * df[col2].std()
clean_mask = mask1 & mask2
rows_removed = (~clean_mask).sum()
hook(rows_removed, "outlier rows removed", name='rows_removed', depends_on=['original_pair'])
print(f"Rows removed as outliers: {rows_removed}")

# Step 3: Recompute correlation on clean data
clean_corr = df.loc[clean_mask, col1].corr(df.loc[clean_mask, col2])
hook(clean_corr, "correlation after outlier removal", name='clean_corr', depends_on=['rows_removed'])
print(f"Correlation after cleaning: {clean_corr:.4f}")

submit({
    "columns": sorted([col1, col2]),
    "original_association": round(float(original_corr), 2),
    "outliers_removed": int(rows_removed),
    "clean_association": round(float(clean_corr), 2)
})
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 3,
    n_steps=5,
    difficulty="HARD",
)


# =============================================================================
# COMPLEX TEMPLATES - Statistical tests & ML (7-10 steps)
# =============================================================================

REGRESSION_MOST_PREDICTIVE = CompositionTemplate(
    name="regression_most_predictive",
    description="Use highest-variance numeric column as target (excluding identifier-like columns), find best predictor via absolute correlation, build OLS regression, report R-squared",
    output_schema='A JSON object with exactly 6 keys: "target" (column name), "best_predictor" (column name), "association_score" (rounded to 2 decimals), "fit_score" (rounded to 2 decimals), "coefficient" (rounded to 2 decimals), and "significance_score" (rounded to 4 decimals). Example: {"target": "price", "best_predictor": "sqft", "association_score": 0.83, "fit_score": 0.6956, "coefficient": 135.2847, "significance_score": 0.0000}',
    code_template="""
# Step 1: Identify numeric columns (exclude index-like columns)
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
hook(len(numeric_cols), "number of numeric columns", name='n_numeric')
print(f"Numeric columns: {len(numeric_cols)}")

# Step 2: Use the column with highest variance as target
variances = df[numeric_cols].var()
target_col = variances.idxmax()
hook(target_col, "target column (highest variance)", name='target_col')
print(f"Target column: {target_col}")

# Step 3: Compute correlations with target
other_cols = [c for c in numeric_cols if c != target_col]
correlations = df[other_cols].corrwith(df[target_col]).abs()
hook(correlations.to_dict(), "correlations with target", name='correlations', depends_on=['target_col'])
print(f"Top correlations:\\n{correlations.sort_values(ascending=False).head()}")

# Step 4: Identify most predictive feature
best_predictor = correlations.idxmax()
best_corr = correlations.max()
hook(best_predictor, "most predictive feature", name='best_predictor', depends_on=['correlations'])
print(f"Best predictor: {best_predictor} (r={best_corr:.4f})")

# Step 5: Prepare data for regression (drop NaN)
reg_data = df[[best_predictor, target_col]].dropna()
X = reg_data[[best_predictor]]
y = reg_data[target_col]
hook(len(reg_data), "samples for regression", name='n_samples', depends_on=['best_predictor'])

# Step 6: Fit OLS regression
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()
r_squared = model.rsquared
hook(r_squared, "R-squared from OLS", name='r_squared', depends_on=['n_samples'])
print(f"R-squared: {r_squared:.4f}")

# Step 7: Get coefficient and p-value
coef = model.params[best_predictor]
p_value = model.pvalues[best_predictor]
hook({"coef": float(coef), "p_value": float(p_value)}, "regression stats", name='reg_stats', depends_on=['r_squared'])

submit({
    "target": target_col,
    "best_predictor": best_predictor,
    "association_score": round(float(best_corr), 2),
    "fit_score": round(float(r_squared), 2),
    "coefficient": round(float(coef), 2),
    "significance_score": round(float(p_value), 4)
})
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 3,
    n_steps=8,
    difficulty="HARD",
    alternative_code_templates=[
        # CV-based target selection
        """
# Step 1: Identify numeric columns (exclude index-like columns)
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
hook(len(numeric_cols), "number of numeric columns", name='n_numeric')
print(f"Numeric columns: {len(numeric_cols)}")

# Step 2: Use the column with highest CV as target
means = df[numeric_cols].mean()
stds = df[numeric_cols].std()
cv = stds / means.abs()
cv = cv.replace([float('inf'), float('-inf')], float('nan')).dropna()
target_col = cv.idxmax()
hook(target_col, "target column (highest CV)", name='target_col')
print(f"Target column: {target_col}")

# Step 3: Compute correlations with target
other_cols = [c for c in numeric_cols if c != target_col]
correlations = df[other_cols].corrwith(df[target_col]).abs()
hook(correlations.to_dict(), "correlations with target", name='correlations', depends_on=['target_col'])
print(f"Top correlations:\\n{correlations.sort_values(ascending=False).head()}")

# Step 4: Identify most predictive feature
best_predictor = correlations.idxmax()
best_corr = correlations.max()
hook(best_predictor, "most predictive feature", name='best_predictor', depends_on=['correlations'])
print(f"Best predictor: {best_predictor} (r={best_corr:.4f})")

# Step 5: Prepare data for regression (drop NaN)
reg_data = df[[best_predictor, target_col]].dropna()
X = reg_data[[best_predictor]]
y = reg_data[target_col]
hook(len(reg_data), "samples for regression", name='n_samples', depends_on=['best_predictor'])

# Step 6: Fit OLS regression
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()
r_squared = model.rsquared
hook(r_squared, "R-squared from OLS", name='r_squared', depends_on=['n_samples'])
print(f"R-squared: {r_squared:.4f}")

# Step 7: Get coefficient and p-value
coef = model.params[best_predictor]
p_value = model.pvalues[best_predictor]
hook({"coef": float(coef), "p_value": float(p_value)}, "regression stats", name='reg_stats', depends_on=['r_squared'])

submit({
    "target": target_col,
    "best_predictor": best_predictor,
    "association_score": round(float(best_corr), 2),
    "fit_score": round(float(r_squared), 2),
    "coefficient": round(float(coef), 2),
    "significance_score": round(float(p_value), 4)
})
""".strip(),
    ],
)

TTEST_DISCOVERED_GROUPS = CompositionTemplate(
    name="ttest_discovered_groups",
    description="Find binary categorical column (if none exists, create binary split from first non-target numeric column's median), perform t-test on highest-variance numeric column (excluding identifier-like columns)",
    output_schema='A JSON object with exactly 9 keys: "target_column", "grouping_column", "group1", "group2", "mean1" (rounded to 2 decimals), "mean2" (rounded to 2 decimals), "test_statistic" (rounded to 2 decimals), "significance_score" (rounded to 4 decimals), and "significant" (boolean, true if significance_score < 0.05). Example: {"target_column": "score", "grouping_column": "gender", "group1": "M", "group2": "F", "mean1": 75.4321, "mean2": 78.1234, "test_statistic": -2.3456, "significance_score": 0.0192, "significant": true}',
    code_template="""
# Step 1: Find numeric column with highest variance
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
variances = df[numeric_cols].var()
target_col = variances.idxmax()
hook(target_col, "target column (highest variance)", name='target_col')
print(f"Target: {target_col}")

# Step 2: Find a binary categorical column (including numeric columns with exactly 2 unique values)
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
binary_col = None
# First check object/category columns
for col in sorted(cat_cols):  # Alphabetical for determinism
    if df[col].nunique() == 2:
        binary_col = col
        break
# Then check numeric columns with exactly 2 unique values (e.g., 0/1 binary)
if binary_col is None:
    for col in sorted(numeric_cols):
        if col != target_col and df[col].nunique() == 2:
            binary_col = col
            break

if binary_col is None:
    # Create binary from numeric if no categorical
    for col in sorted(numeric_cols):  # Alphabetical for determinism
        if col != target_col:
            median_val = df[col].median()
            df['_binary_group'] = (df[col] > median_val).map({True: 'high', False: 'low'})
            binary_col = '_binary_group'
            break

if binary_col is None:
    submit({"error": "No binary grouping column available"})
else:
    hook(binary_col, "binary grouping column", name='binary_col')
    print(f"Grouping by: {binary_col}")

    # Step 3: Get the two groups (sorted for determinism)
    raw_groups = list(df[binary_col].dropna().unique())
    sorted_groups = sorted(raw_groups, key=str)  # Sort by string repr, keep original values
    group1_val, group2_val = sorted_groups[0], sorted_groups[1]
    group1_name, group2_name = str(group1_val), str(group2_val)
    hook([group1_name, group2_name], "group names", name='group_names', depends_on=['binary_col'])

    # Step 4: Extract data for each group
    group1_data = df[df[binary_col] == group1_val][target_col].dropna()
    group2_data = df[df[binary_col] == group2_val][target_col].dropna()
    hook({"group1_n": len(group1_data), "group2_n": len(group2_data)}, "group sizes", name='group_sizes', depends_on=['group_names'])
    print(f"Group sizes: {len(group1_data)} vs {len(group2_data)}")

    # Step 5: Compute group means
    mean1, mean2 = group1_data.mean(), group2_data.mean()
    hook({"mean1": float(mean1), "mean2": float(mean2)}, "group means", name='group_means', depends_on=['group_sizes'])
    print(f"Means: {mean1:.4f} vs {mean2:.4f}")

    # Step 6: Perform t-test
    t_stat, p_value = scipy.stats.ttest_ind(group1_data, group2_data)
    hook({"t_stat": float(t_stat), "p_value": float(p_value)}, "t-test results", name='ttest', depends_on=['group_means'])
    print(f"T-test: t={t_stat:.4f}, p={p_value:.6f}")

    # Step 7: Determine significance
    is_significant = p_value < 0.05
    hook(is_significant, "significance at alpha=0.05", name='is_significant', depends_on=['ttest'])

    submit({
        "target_column": target_col,
        "grouping_column": binary_col,
        "group1": str(group1_name),
        "group2": str(group2_name),
        "mean1": round(float(mean1), 2),
        "mean2": round(float(mean2), 2),
        "test_statistic": round(float(t_stat), 2),
        "significance_score": round(float(p_value), 4),
        "significant": is_significant
    })
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 2,
    n_steps=8,
    difficulty="HARD",
    alternative_code_templates=[
        # CV-based target selection
        """
# Step 1: Find numeric column with highest coefficient of variation
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
means = df[numeric_cols].mean()
stds = df[numeric_cols].std()
cv = stds / means.abs()
cv = cv.replace([float('inf'), float('-inf')], float('nan')).dropna()
target_col = cv.idxmax()
hook(target_col, "target column (highest CV)", name='target_col')
print(f"Target: {target_col}")

# Step 2: Find a binary categorical column (including numeric columns with exactly 2 unique values)
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
binary_col = None
# First check object/category columns
for col in sorted(cat_cols):  # Alphabetical for determinism
    if df[col].nunique() == 2:
        binary_col = col
        break
# Then check numeric columns with exactly 2 unique values (e.g., 0/1 binary)
if binary_col is None:
    for col in sorted(numeric_cols):
        if col != target_col and df[col].nunique() == 2:
            binary_col = col
            break

if binary_col is None:
    # Create binary from numeric if no categorical
    for col in sorted(numeric_cols):  # Alphabetical for determinism
        if col != target_col:
            median_val = df[col].median()
            df['_binary_group'] = (df[col] > median_val).map({True: 'high', False: 'low'})
            binary_col = '_binary_group'
            break

if binary_col is None:
    submit({"error": "No binary grouping column available"})
else:
    hook(binary_col, "binary grouping column", name='binary_col')
    print(f"Grouping by: {binary_col}")

    # Step 3: Get the two groups (sorted for determinism)
    raw_groups = list(df[binary_col].dropna().unique())
    sorted_groups = sorted(raw_groups, key=str)  # Sort by string repr, keep original values
    group1_val, group2_val = sorted_groups[0], sorted_groups[1]
    group1_name, group2_name = str(group1_val), str(group2_val)
    hook([group1_name, group2_name], "group names", name='group_names', depends_on=['binary_col'])

    # Step 4: Extract data for each group
    group1_data = df[df[binary_col] == group1_val][target_col].dropna()
    group2_data = df[df[binary_col] == group2_val][target_col].dropna()
    hook({"group1_n": len(group1_data), "group2_n": len(group2_data)}, "group sizes", name='group_sizes', depends_on=['group_names'])
    print(f"Group sizes: {len(group1_data)} vs {len(group2_data)}")

    # Step 5: Compute group means
    mean1, mean2 = group1_data.mean(), group2_data.mean()
    hook({"mean1": float(mean1), "mean2": float(mean2)}, "group means", name='group_means', depends_on=['group_sizes'])
    print(f"Means: {mean1:.4f} vs {mean2:.4f}")

    # Step 6: Perform t-test
    t_stat, p_value = scipy.stats.ttest_ind(group1_data, group2_data)
    hook({"t_stat": float(t_stat), "p_value": float(p_value)}, "t-test results", name='ttest', depends_on=['group_means'])
    print(f"T-test: t={t_stat:.4f}, p={p_value:.6f}")

    # Step 7: Determine significance
    is_significant = p_value < 0.05
    hook(is_significant, "significance at alpha=0.05", name='is_significant', depends_on=['ttest'])

    submit({
        "target_column": target_col,
        "grouping_column": binary_col,
        "group1": str(group1_name),
        "group2": str(group2_name),
        "mean1": round(float(mean1), 2),
        "mean2": round(float(mean2), 2),
        "test_statistic": round(float(t_stat), 2),
        "significance_score": round(float(p_value), 4),
        "significant": is_significant
    })
""".strip(),
    ],
)

BOOTSTRAP_CI_DISCOVERED = CompositionTemplate(
    name="bootstrap_ci_discovered",
    description="Bootstrap 95% CI (1000 iterations, seed=42) for the mean of the column with highest absolute skewness (excluding identifier-like columns)",
    output_schema='A JSON object with exactly 7 keys: "column" (name of most skewed column), "skewness" (rounded to 2 decimals), "mean" (rounded to 2 decimals), "ci_lower" (lower bound of 95% CI, rounded to 2 decimals), "ci_upper" (upper bound, rounded to 2 decimals), "std_error" (bootstrap standard error, rounded to 2 decimals), and "n_iterations" (integer, the number of resamples). Example: {"column": "income", "skewness": 2.3456, "mean": 50000.1234, "ci_lower": 48500.5678, "ci_upper": 51500.9012, "std_error": 750.3456, "n_iterations": 1000}',
    code_template="""
# Step 1: Identify numeric columns
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
hook(len(numeric_cols), "number of numeric columns", name='n_numeric')

# Step 2: Compute skewness for each column
skewness = {}
for col in numeric_cols:
    skew_val = df[col].skew()
    if not np.isnan(skew_val):
        skewness[col] = abs(skew_val)

hook(skewness, "absolute skewness per column", name='skewness')
print(f"Skewness values:\\n{sorted(skewness.items(), key=lambda x: -x[1])[:5]}")

# Step 3: Find column with highest absolute skewness (alphabetical tiebreaker)
target_col = sorted(skewness.items(), key=lambda x: (-x[1], x[0]))[0][0]
hook(target_col, "most skewed column", name='target_col', depends_on=['skewness'])
print(f"Most skewed: {target_col} (skewness={skewness[target_col]:.4f})")

# Step 4: Get clean data
data = df[target_col].dropna().values
n = len(data)
hook(n, "sample size", name='sample_size', depends_on=['target_col'])
print(f"Sample size: {n}")

# Step 5: Compute original mean
original_mean = data.mean()
hook(original_mean, "original mean", name='original_mean', depends_on=['sample_size'])
print(f"Original mean: {original_mean:.4f}")

# Step 6: Bootstrap resampling (1000 iterations)
rng = np.random.default_rng(42)
n_bootstrap = 1000
bootstrap_means = []
for _ in range(n_bootstrap):
    sample = rng.choice(data, size=n, replace=True)
    bootstrap_means.append(sample.mean())

bootstrap_means = np.array(bootstrap_means)
hook({"n_bootstrap": n_bootstrap, "mean_of_means": float(bootstrap_means.mean())}, "bootstrap info", name='bootstrap_info', depends_on=['original_mean'])

# Step 7: Compute 95% CI
ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)
hook({"ci_lower": float(ci_lower), "ci_upper": float(ci_upper)}, "95% CI", name='confidence_interval', depends_on=['bootstrap_info'])
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Step 8: Compute standard error
se = bootstrap_means.std()
hook(se, "bootstrap standard error", name='std_error', depends_on=['confidence_interval'])

    submit({
        "column": target_col,
        "skewness": round(float(skewness[target_col]), 2),
        "mean": round(float(original_mean), 2),
        "ci_lower": round(float(ci_lower), 2),
        "ci_upper": round(float(ci_upper), 2),
        "std_error": round(float(se), 2),
        "n_iterations": n_bootstrap
    })
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 2,
    n_steps=9,
    difficulty="VERY_HARD",
)

ANOVA_DISCOVERED_GROUPS = CompositionTemplate(
    name="anova_discovered_groups",
    description="Find first categorical column with 3-10 groups (each with n >= 10 samples), perform one-way ANOVA on highest-variance numeric column (excluding identifier-like columns)",
    output_schema='A JSON object with exactly 11 keys: "target_column", "grouping_column", "n_groups" (integer), "test_statistic" (rounded to 2 decimals), "significance_score" (rounded to 4 decimals), "significant" (boolean), "best_group" (category with highest mean), "best_mean" (rounded to 2 decimals), "worst_group" (category with lowest mean), "worst_mean" (rounded to 2 decimals), and "effect_size" (rounded to 2 decimals). Example: {"target_column": "sales", "grouping_column": "region", "n_groups": 4, "test_statistic": 15.23, "significance_score": 0.0000, "significant": true, "best_group": "West", "best_mean": 1234.5678, "worst_group": "East", "worst_mean": 890.1234, "effect_size": 0.1523}',
    code_template="""
# Step 1: Find numeric column with highest variance
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
variances = df[numeric_cols].var()
target_col = variances.idxmax()
hook(target_col, "target column (highest variance)", name='target_col')
print(f"Target: {target_col}")

# Step 2: Find categorical column with 3-10 unique values
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
group_col = None
for col in cat_cols:
    n_unique = df[col].nunique()
    if 3 <= n_unique <= 10:
        group_col = col
        break

if group_col is None:
    cat_cardinalities = {col: df[col].nunique() for col in cat_cols[:5]}
    submit({"error": f"No categorical column with 3-10 groups. Found cardinalities: {cat_cardinalities}"})
else:
    hook(group_col, "grouping column", name='group_col')
    print(f"Grouping by: {group_col}")

    # Step 3: Get group names and sizes - require minimum 10 samples per group
    group_counts = df[group_col].value_counts()
    valid_groups = group_counts[group_counts >= 10]
    n_groups = len(valid_groups)

    if n_groups < 3:
        submit({"error": f"Need 3+ groups with ≥10 samples. Found {n_groups} valid groups (sizes: {group_counts.to_dict()})"})
    else:
        hook({"n_groups": n_groups, "counts": valid_groups.to_dict()}, "group info", name='group_info', depends_on=['group_col'])
        print(f"Valid groups with ≥10 samples: {n_groups}")

        # Step 4: Compute group means (only for valid groups)
        df_valid = df[df[group_col].isin(valid_groups.index)]
        group_means = df_valid.groupby(group_col)[target_col].mean()
        hook(group_means.to_dict(), "group means", name='group_means', depends_on=['group_info'])
        print(f"Group means:\\n{group_means.sort_values(ascending=False)}")

        # Step 5: Prepare data for ANOVA
        groups_data = [df_valid[df_valid[group_col] == g][target_col].dropna().values for g in valid_groups.index]
        min_group_size = min(len(g) for g in groups_data)
        hook(min_group_size, "minimum group size", name='min_size', depends_on=['group_means'])

        # Step 6: Perform one-way ANOVA
        f_stat, p_value = scipy.stats.f_oneway(*groups_data)
        hook({"f_stat": float(f_stat), "p_value": float(p_value)}, "ANOVA results", name='anova', depends_on=['min_size'])
        print(f"ANOVA: F={f_stat:.4f}, p={p_value:.6f}")

        # Step 7: Identify best and worst groups
        best_group = group_means.idxmax()
        worst_group = group_means.idxmin()
        hook({"best": str(best_group), "worst": str(worst_group)}, "extreme groups", name='extremes', depends_on=['anova'])

        # Step 8: Effect size (eta-squared)
        ss_between = sum(len(g) * (g.mean() - df_valid[target_col].mean())**2 for g in groups_data)
        ss_total = ((df_valid[target_col] - df_valid[target_col].mean())**2).sum()
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        hook(eta_squared, "eta-squared effect size", name='effect_size', depends_on=['extremes'])

    submit({
        "target_column": target_col,
        "grouping_column": group_col,
        "n_groups": n_groups,
        "test_statistic": round(float(f_stat), 2),
        "significance_score": round(float(p_value), 4),
        "significant": p_value < 0.05,
        "best_group": str(best_group),
        "best_mean": round(float(group_means[best_group]), 2),
        "worst_group": str(worst_group),
        "worst_mean": round(float(group_means[worst_group]), 2),
        "effect_size": round(float(eta_squared), 2)
    })
""".strip(),
    output_type="dict",
    applicable_when=lambda p: (
        _count_numeric_cols(p) >= 1 and
        _has_moderate_cardinality_categorical(p, min_groups=3, max_groups=10)
    ),
    n_steps=9,
    difficulty="VERY_HARD",
    alternative_code_templates=[
        # Alternative: Exclude the highest-variance column (likely target) and use second-highest
        """
# Step 1: Find numeric column with second-highest variance (excluding likely target)
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
variances = df[numeric_cols].var()
# Exclude highest variance (likely the target variable)
target_candidate = variances.idxmax()
remaining_vars = variances.drop(target_candidate)
target_col = remaining_vars.idxmax()
hook(target_col, "target column (second-highest variance, excluding likely target)", name='target_col')
print(f"Target: {target_col} (excluded {target_candidate} as likely target)")

# Step 2: Find categorical column with 3-10 unique values
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
group_col = None
for col in cat_cols:
    n_unique = df[col].nunique()
    if 3 <= n_unique <= 10:
        group_col = col
        break

if group_col is None:
    cat_cardinalities = {col: df[col].nunique() for col in cat_cols[:5]}
    submit({"error": f"No categorical column with 3-10 groups. Found cardinalities: {cat_cardinalities}"})
else:
    hook(group_col, "grouping column", name='group_col')
    print(f"Grouping by: {group_col}")

    # Step 3: Get group names and sizes - require minimum 10 samples per group
    group_counts = df[group_col].value_counts()
    valid_groups = group_counts[group_counts >= 10]
    n_groups = len(valid_groups)

    if n_groups < 3:
        submit({"error": f"Need 3+ groups with ≥10 samples. Found {n_groups} valid groups"})
    else:
        hook({"n_groups": n_groups, "counts": valid_groups.to_dict()}, "group info", name='group_info', depends_on=['group_col'])
        print(f"Valid groups with ≥10 samples: {n_groups}")

        # Step 4: Compute group means (only for valid groups)
        df_valid = df[df[group_col].isin(valid_groups.index)]
        group_means = df_valid.groupby(group_col)[target_col].mean()
        hook(group_means.to_dict(), "group means", name='group_means', depends_on=['group_info'])

        # Step 5: Prepare data for ANOVA
        groups_data = [df_valid[df_valid[group_col] == g][target_col].dropna().values for g in valid_groups.index]
        min_group_size = min(len(g) for g in groups_data)
        hook(min_group_size, "minimum group size", name='min_size', depends_on=['group_means'])

        # Step 6: Perform one-way ANOVA
        f_stat, p_value = scipy.stats.f_oneway(*groups_data)
        hook({"f_stat": float(f_stat), "p_value": float(p_value)}, "ANOVA results", name='anova', depends_on=['min_size'])

        # Step 7: Identify best and worst groups
        best_group = group_means.idxmax()
        worst_group = group_means.idxmin()
        hook({"best": str(best_group), "worst": str(worst_group)}, "extreme groups", name='extremes', depends_on=['anova'])

        # Step 8: Effect size (eta-squared)
        ss_between = sum(len(g) * (g.mean() - df_valid[target_col].mean())**2 for g in groups_data)
        ss_total = ((df_valid[target_col] - df_valid[target_col].mean())**2).sum()
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        hook(eta_squared, "eta-squared effect size", name='effect_size', depends_on=['extremes'])

    submit({
        "target_column": target_col,
        "grouping_column": group_col,
        "n_groups": n_groups,
        "test_statistic": round(float(f_stat), 2),
        "significance_score": round(float(p_value), 4),
        "significant": p_value < 0.05,
        "best_group": str(best_group),
        "best_mean": round(float(group_means[best_group]), 2),
        "worst_group": str(worst_group),
        "worst_mean": round(float(group_means[worst_group]), 2),
        "effect_size": round(float(eta_squared), 2)
    })
""".strip(),
    ],
)

MULTIPLE_REGRESSION_TOP_PREDICTORS = CompositionTemplate(
    name="multiple_regression_top_predictors",
    description="Use highest-variance numeric column as target (excluding identifier-like columns), find top 3 predictors via absolute correlation, build OLS multiple regression, report adjusted R-squared",
    output_schema='A JSON object with exactly 5 keys: "target" (column name), "predictors" (list of 3 column names, alphabetically sorted), "fit_score" (rounded to 2 decimals), "fit_score_adj" (rounded to 2 decimals), and "n_influential" (integer count of predictors with significance_score < 0.05). Example: {"target": "price", "predictors": ["age", "bedrooms", "sqft"], "fit_score": 0.7234, "fit_score_adj": 0.7156, "n_influential": 2}',
    code_template="""
# Step 1: Identify numeric columns
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
hook(len(numeric_cols), "number of numeric columns", name='n_numeric')
print(f"Numeric columns: {len(numeric_cols)}")

if len(numeric_cols) < 4:
    submit({"error": "Need at least 4 numeric columns for multiple regression"})
else:
    # Step 2: Use the column with highest variance as target
    variances = df[numeric_cols].var()
    target_col = variances.idxmax()
    hook(target_col, "target column (highest variance)", name='target_col')
    print(f"Target: {target_col}")

    # Step 3: Compute correlations with target
    other_cols = [c for c in numeric_cols if c != target_col]
    correlations = df[other_cols].corrwith(df[target_col]).abs()
    hook(correlations.to_dict(), "correlations with target", name='correlations', depends_on=['target_col'])

    # Step 4: Select top 3 predictors
    top_3 = correlations.nlargest(3).index.tolist()
    hook(top_3, "top 3 predictors", name='top_predictors', depends_on=['correlations'])
    print(f"Top 3 predictors: {top_3}")

    # Step 5: Check for multicollinearity
    predictor_corr = df[top_3].corr()
    max_predictor_corr = predictor_corr.where(np.triu(np.ones(predictor_corr.shape), k=1).astype(bool)).stack().abs().max()
    hook(float(max_predictor_corr), "max predictor intercorrelation", name='multicollinearity', depends_on=['top_predictors'])
    print(f"Max predictor intercorrelation: {max_predictor_corr:.4f}")

    # Step 6: Prepare regression data
    reg_data = df[[target_col] + top_3].dropna()
    X = reg_data[top_3]
    y = reg_data[target_col]
    hook(len(reg_data), "samples for regression", name='n_samples', depends_on=['multicollinearity'])
    print(f"Samples: {len(reg_data)}")

    # Step 7: Fit OLS regression
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    hook({"r_squared": float(model.rsquared), "adj_r_squared": float(model.rsquared_adj)}, "model fit", name='model_fit', depends_on=['n_samples'])
    print(f"R-squared: {model.rsquared:.4f}, Adjusted: {model.rsquared_adj:.4f}")

    # Step 8: Count significant predictors
    n_significant = sum(1 for col in top_3 if model.pvalues[col] < 0.05)
    hook(n_significant, "significant predictors at alpha=0.05", name='n_significant', depends_on=['model_fit'])

    submit({
        "target": target_col,
        "predictors": sorted(top_3),
        "fit_score": round(float(model.rsquared), 2),
        "fit_score_adj": round(float(model.rsquared_adj), 2),
        "n_influential": n_significant
    })
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 4,
    n_steps=9,
    difficulty="VERY_HARD",
)


# =============================================================================
# INTRO STATISTICS - Basic hypothesis testing and descriptive stats
# =============================================================================

CHI_SQUARED_INDEPENDENCE = CompositionTemplate(
    name="chi_squared_independence",
    description="Test independence between first two categorical columns (each with 2-10 unique values) using chi-squared test at alpha=0.05",
    output_schema='A JSON object with exactly 7 keys: "column1" (first categorical column), "column2" (second categorical column), "test_statistic" (rounded to 2 decimals), "significance_score" (rounded to 4 decimals), "df" (integer), "expected_min" (minimum expected frequency rounded to 2 decimals), and "independent" (boolean, true if significance_score >= 0.05). Example: {"column1": "gender", "column2": "product", "test_statistic": 15.23, "significance_score": 0.0043, "df": 4, "expected_min": 5.23, "independent": false}',
    code_template="""
# Step 1: Identify categorical columns with reasonable cardinality
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
suitable_cats = [c for c in cat_cols if 2 <= df[c].nunique() <= 10]
hook(suitable_cats, "suitable categorical columns", name='suitable_cats')

if len(suitable_cats) < 2:
    submit({"error": "Need at least 2 categorical columns with 2-10 unique values"})
else:
    col1, col2 = suitable_cats[0], suitable_cats[1]
    hook([col1, col2], "selected columns for chi-squared test", name='selected_cols')
    print(f"Testing independence: {col1} vs {col2}")

    # Step 2: Create contingency table
    contingency = pd.crosstab(df[col1], df[col2])
    hook(contingency.shape, "contingency table shape", name='table_shape', depends_on=['selected_cols'])
    print(f"Contingency table: {contingency.shape[0]} x {contingency.shape[1]}")

    # Step 3: Perform chi-squared test
    chi2, p_value, dof, expected = scipy.stats.chi2_contingency(contingency)
    hook({"chi2": float(chi2), "p_value": float(p_value), "dof": int(dof)}, "chi-squared results", name='chi2_results', depends_on=['table_shape'])
    print(f"Chi-squared: {chi2:.4f}, p-value: {p_value:.6f}, dof: {dof}")

    # Step 4: Check expected frequencies (should be >= 5 for validity)
    min_expected = expected.min()
    hook(min_expected, "minimum expected frequency", name='min_expected', depends_on=['chi2_results'])
    print(f"Min expected frequency: {min_expected:.2f}")

    # Step 5: Determine independence
    is_independent = p_value >= 0.05
    hook(is_independent, "independence at alpha=0.05", name='is_independent', depends_on=['min_expected'])

    submit({
        "column1": col1,
        "column2": col2,
        "test_statistic": round(float(chi2), 2),
        "significance_score": round(float(p_value), 4),
        "df": int(dof),
        "expected_min": round(float(min_expected), 2),
        "independent": is_independent
    })
""".strip(),
    output_type="dict",
    applicable_when=lambda p: sum(
        1 for c in p.get("columns", {}).values() if c.get("type") == "categorical"
    )
    >= 2,
    n_steps=6,
    difficulty="MEDIUM",
)

MANN_WHITNEY_U_TEST = CompositionTemplate(
    name="mann_whitney_u_test",
    description="Find binary categorical column (if none, create median split from first non-target numeric), perform Mann-Whitney U test on most-skewed numeric column (excluding identifier-like columns)",
    output_schema='A JSON object with exactly 8 keys: "target_column", "grouping_column", "group1", "group2", "median1" (rounded to 2 decimals), "median2" (rounded to 2 decimals), "test_statistic" (rounded to 2 decimals), and "significance_score" (rounded to 4 decimals). Example: {"target_column": "score", "grouping_column": "treatment", "group1": "control", "group2": "experimental", "median1": 72.5000, "median2": 78.0000, "test_statistic": 1234.50, "significance_score": 0.0345}',
    code_template="""
# Step 1: Find most skewed numeric column (non-parametric tests suit skewed data)
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
skewness = {col: abs(df[col].skew()) for col in numeric_cols if not df[col].isna().all()}
# Alphabetical tiebreaker for determinism
target_col = sorted(skewness.items(), key=lambda x: (-x[1], x[0]))[0][0]
hook({"target": target_col, "skewness": skewness[target_col]}, "target column (most skewed)", name='target_col')
print(f"Target: {target_col} (skewness: {skewness[target_col]:.4f})")

# Step 2: Find binary grouping column (including numeric columns with exactly 2 unique values)
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
binary_col = None
# First check object/category columns
for col in sorted(cat_cols):  # Alphabetical for determinism
    if df[col].nunique() == 2:
        binary_col = col
        break
# Then check numeric columns with exactly 2 unique values (e.g., 0/1 binary)
if binary_col is None:
    for col in sorted(numeric_cols):
        if col != target_col and df[col].nunique() == 2:
            binary_col = col
            break

if binary_col is None:
    # Create binary from median split
    other_cols = sorted([c for c in numeric_cols if c != target_col])  # Alphabetical for determinism
    other_numeric = other_cols[0] if other_cols else target_col
    df['_binary_group'] = (df[other_numeric] > df[other_numeric].median()).map({True: 'high', False: 'low'})
    binary_col = '_binary_group'

hook(binary_col, "grouping column", name='binary_col')
print(f"Grouping by: {binary_col}")

# Step 3: Extract groups (sorted for determinism)
raw_groups = list(df[binary_col].dropna().unique())
# Sort by string representation for determinism, but keep original values for filtering
sorted_groups = sorted(raw_groups, key=str)
g1_val, g2_val = sorted_groups[0], sorted_groups[1]
g1_name, g2_name = str(g1_val), str(g2_val)
hook([g1_name, g2_name], "group names", name='groups', depends_on=['binary_col'])

g1_data = df[df[binary_col] == g1_val][target_col].dropna()
g2_data = df[df[binary_col] == g2_val][target_col].dropna()
hook({"n1": len(g1_data), "n2": len(g2_data)}, "group sizes", name='sizes', depends_on=['groups'])

# Step 4: Compute medians
med1, med2 = g1_data.median(), g2_data.median()
hook({"median1": float(med1), "median2": float(med2)}, "group medians", name='medians', depends_on=['sizes'])
print(f"Medians: {med1:.4f} vs {med2:.4f}")

# Step 5: Mann-Whitney U test
u_stat, p_value = scipy.stats.mannwhitneyu(g1_data, g2_data, alternative='two-sided')
hook({"u_stat": float(u_stat), "p_value": float(p_value)}, "Mann-Whitney results", name='mw_results', depends_on=['medians'])
print(f"U-statistic: {u_stat:.2f}, p-value: {p_value:.6f}")

submit({
    "target_column": target_col,
    "grouping_column": binary_col,
    "group1": g1_name,
    "group2": g2_name,
    "median1": round(float(med1), 2),
    "median2": round(float(med2), 2),
    "test_statistic": round(float(u_stat), 2),
    "significance_score": round(float(p_value), 4)
})
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 1,
    n_steps=6,
    difficulty="MEDIUM",
)

SPEARMAN_RANK_CORRELATION = CompositionTemplate(
    name="spearman_rank_correlation",
    description="Find strongest absolute monotonic (Spearman rank) correlation between numeric columns (excluding identifier-like columns); interpretation thresholds: >=0.7 strong, >=0.4 moderate, >=0.2 weak, else negligible",
    output_schema='A JSON object with exactly 4 keys: "columns" (list of 2 column names, alphabetically sorted), "association_score" (rounded to 2 decimals), "significance_score" (rounded to 4 decimals), and "interpretation" (string: "strong", "moderate", "weak", or "negligible"). Example: {"columns": ["age", "income"], "association_score": 0.72, "significance_score": 0.0000, "interpretation": "strong"}',
    code_template="""
# Step 1: Get numeric columns
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
hook(len(numeric_cols), "number of numeric columns", name='n_cols')
print(f"Numeric columns: {len(numeric_cols)}")

# Step 2: Compute Spearman correlation matrix
spearman_matrix = df[numeric_cols].corr(method='spearman').abs()
np.fill_diagonal(spearman_matrix.values, 0)
hook(spearman_matrix.shape, "Spearman correlation matrix computed", name='corr_matrix')

# Step 3: Find strongest pair
max_idx = spearman_matrix.stack().idxmax()
col1, col2 = max_idx
hook([col1, col2], "strongest Spearman pair", name='max_pair', depends_on=['corr_matrix'])
print(f"Strongest monotonic correlation: {col1} vs {col2}")

# Step 4: Compute exact Spearman with p-value
rho, p_value = scipy.stats.spearmanr(df[col1].dropna(), df[col2].dropna())
hook({"rho": float(rho), "p_value": float(p_value)}, "Spearman results", name='spearman', depends_on=['max_pair'])
print(f"Spearman rho: {rho:.4f}, p-value: {p_value:.6f}")

# Step 5: Interpret strength
abs_rho = abs(rho)
if abs_rho >= 0.7:
    interpretation = "strong"
elif abs_rho >= 0.4:
    interpretation = "moderate"
elif abs_rho >= 0.2:
    interpretation = "weak"
else:
    interpretation = "negligible"
hook(interpretation, "correlation interpretation", name='interpretation', depends_on=['spearman'])

submit({
    "columns": sorted([col1, col2]),
    "association_score": round(float(rho), 2),
    "significance_score": round(float(p_value), 4),
    "interpretation": interpretation
})
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 3,
    n_steps=6,
    difficulty="MEDIUM",
)

COEFFICIENT_OF_VARIATION = CompositionTemplate(
    name="coefficient_of_variation",
    description="Find the column with highest relative variability (coefficient of variation, excluding identifier-like columns)",
    output_schema='A JSON object with exactly 4 keys: "column" (name of column with highest CV), "cv" (coefficient of variation as percentage rounded to 2 decimals), "mean" (rounded to 2 decimals), and "std" (rounded to 2 decimals). Example: {"column": "price", "cv": 45.23, "mean": 1234.5678, "std": 558.4567}',
    code_template="""
# Step 1: Get numeric columns with non-zero means
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
cv_values = {}

for col in numeric_cols:
    mean_val = df[col].mean()
    std_val = df[col].std()
    if mean_val != 0 and std_val > 0:
        cv_values[col] = (std_val / abs(mean_val)) * 100

if not cv_values:
    submit({"error": "No columns with non-zero mean found for CV calculation"})
else:
    hook(cv_values, "coefficient of variation per column", name='cv_values')
    print("Coefficient of Variation (%):")
    for col, cv in sorted(cv_values.items(), key=lambda x: -x[1])[:5]:
        print(f"  {col}: {cv:.2f}%")

    # Step 2: Find column with highest CV (alphabetical tiebreaker)
    max_cv_col = sorted(cv_values.items(), key=lambda x: (-x[1], x[0]))[0][0]
    hook(max_cv_col, "column with highest CV", name='max_cv_col', depends_on=['cv_values'])
    print(f"Highest CV: {max_cv_col} ({cv_values[max_cv_col]:.2f}%)")

    # Step 3: Get mean and std for that column
    col_mean = df[max_cv_col].mean()
    col_std = df[max_cv_col].std()
    hook({"mean": float(col_mean), "std": float(col_std)}, "column statistics", name='col_stats', depends_on=['max_cv_col'])

    submit({
        "column": max_cv_col,
        "cv": round(cv_values[max_cv_col], 2),
        "mean": round(float(col_mean), 2),
        "std": round(float(col_std), 2)
    })
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 2,
    n_steps=4,
    difficulty="EASY",
)

IQR_OUTLIER_ANALYSIS = CompositionTemplate(
    name="iqr_outlier_analysis",
    description="Detect outliers using IQR method (1.5*IQR beyond quartiles) on highest-variance column (excluding identifier-like columns)",
    output_schema='A JSON object with exactly 7 keys: "column" (analyzed column), "q1" (25th percentile rounded to 2 decimals), "q3" (75th percentile rounded to 2 decimals), "iqr" (rounded to 2 decimals), "lower_fence" (rounded to 2 decimals), "upper_fence" (rounded to 2 decimals), and "n_outliers" (integer count). Example: {"column": "salary", "q1": 45000.0000, "q3": 75000.0000, "iqr": 30000.0000, "lower_fence": 0.0000, "upper_fence": 120000.0000, "n_outliers": 23}',
    code_template="""
# Step 1: Find column with highest variance
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
variances = df[numeric_cols].var()
target_col = variances.idxmax()
hook(target_col, "target column (highest variance)", name='target_col')
print(f"Analyzing: {target_col}")

# Step 2: Compute quartiles
q1 = df[target_col].quantile(0.25)
q3 = df[target_col].quantile(0.75)
iqr = q3 - q1
hook({"q1": float(q1), "q3": float(q3), "iqr": float(iqr)}, "quartile info", name='quartiles', depends_on=['target_col'])
print(f"Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")

# Step 3: Calculate fences
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
hook({"lower": float(lower_fence), "upper": float(upper_fence)}, "fences", name='fences', depends_on=['quartiles'])
print(f"Fences: [{lower_fence:.4f}, {upper_fence:.4f}]")

# Step 4: Count outliers
data = df[target_col].dropna()
outliers = ((data < lower_fence) | (data > upper_fence)).sum()
hook(outliers, "outlier count", name='n_outliers', depends_on=['fences'])
print(f"Outliers: {outliers} ({100*outliers/len(data):.2f}%)")

submit({
    "column": target_col,
    "q1": round(float(q1), 2),
    "q3": round(float(q3), 2),
    "iqr": round(float(iqr), 2),
    "lower_fence": round(float(lower_fence), 2),
    "upper_fence": round(float(upper_fence), 2),
    "n_outliers": int(outliers)
})
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 1,
    n_steps=5,
    difficulty="EASY",
    alternative_code_templates=[
        # CV-based alternative
        """
# Step 1: Find column with highest coefficient of variation
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
means = df[numeric_cols].mean()
stds = df[numeric_cols].std()
cv = stds / means.abs()
cv = cv.replace([float('inf'), float('-inf')], float('nan')).dropna()
target_col = cv.idxmax()
hook(target_col, "target column (highest CV)", name='target_col')
print(f"Analyzing: {target_col}")

# Step 2: Compute quartiles
q1 = df[target_col].quantile(0.25)
q3 = df[target_col].quantile(0.75)
iqr = q3 - q1
hook({"q1": float(q1), "q3": float(q3), "iqr": float(iqr)}, "quartile info", name='quartiles', depends_on=['target_col'])
print(f"Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")

# Step 3: Calculate fences
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
hook({"lower": float(lower_fence), "upper": float(upper_fence)}, "fences", name='fences', depends_on=['quartiles'])
print(f"Fences: [{lower_fence:.4f}, {upper_fence:.4f}]")

# Step 4: Count outliers
data = df[target_col].dropna()
outliers = ((data < lower_fence) | (data > upper_fence)).sum()
hook(outliers, "outlier count", name='n_outliers', depends_on=['fences'])
print(f"Outliers: {outliers} ({100*outliers/len(data):.2f}%)")

submit({
    "column": target_col,
    "q1": round(float(q1), 2),
    "q3": round(float(q3), 2),
    "iqr": round(float(iqr), 2),
    "lower_fence": round(float(lower_fence), 2),
    "upper_fence": round(float(upper_fence), 2),
    "n_outliers": int(outliers)
})
""".strip(),
    ],
)

PERCENTILE_RANKING = CompositionTemplate(
    name="percentile_ranking",
    description="Find the value at specific percentiles for the column with highest range (excluding identifier-like columns)",
    output_schema='A JSON object with exactly 6 keys: "column" (analyzed column), "p10" (10th percentile rounded to 2 decimals), "p25" (25th percentile), "p50" (median), "p75" (75th percentile), and "p90" (90th percentile). Example: {"column": "income", "p10": 25000.0000, "p25": 40000.0000, "p50": 55000.0000, "p75": 75000.0000, "p90": 100000.0000}',
    code_template="""
# Step 1: Find column with largest range
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
ranges = {col: df[col].max() - df[col].min() for col in numeric_cols}
# Alphabetical tiebreaker for determinism
target_col = sorted(ranges.items(), key=lambda x: (-x[1], x[0]))[0][0]
hook({"column": target_col, "range": ranges[target_col]}, "column with largest range", name='target_col')
print(f"Analyzing: {target_col} (range: {ranges[target_col]:.4f})")

# Step 2: Compute percentiles
percentiles = [10, 25, 50, 75, 90]
values = {f"p{p}": df[target_col].quantile(p/100) for p in percentiles}
hook(values, "percentile values", name='percentiles', depends_on=['target_col'])
print("Percentiles:")
for k, v in values.items():
    print(f"  {k}: {v:.4f}")

submit({
    "column": target_col,
    "p10": round(float(values["p10"]), 2),
    "p25": round(float(values["p25"]), 2),
    "p50": round(float(values["p50"]), 2),
    "p75": round(float(values["p75"]), 2),
    "p90": round(float(values["p90"]), 2)
})
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 1,
    n_steps=3,
    difficulty="EASY",
)

DESCRIPTIVE_SUMMARY = CompositionTemplate(
    name="descriptive_summary",
    description="Compute comprehensive descriptive statistics for the most variable column (excluding identifier-like columns)",
    output_schema='A JSON object with exactly 9 keys: "column", "count" (integer), "mean" (rounded to 2 decimals), "std", "min", "max", "median", "skewness", and "kurtosis" (all rounded to 2 decimals). Example: {"column": "age", "count": 1000, "mean": 35.4567, "std": 12.3456, "min": 18.0000, "max": 85.0000, "median": 34.0000, "skewness": 0.5678, "kurtosis": -0.2345}',
    code_template="""
# Step 1: Find column with highest variance
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
variances = df[numeric_cols].var()
target_col = variances.idxmax()
hook(target_col, "target column", name='target_col')
print(f"Summarizing: {target_col}")

# Step 2: Compute all descriptive statistics
data = df[target_col].dropna()
stats_dict = {
    "count": len(data),
    "mean": data.mean(),
    "std": data.std(),
    "min": data.min(),
    "max": data.max(),
    "median": data.median(),
    "skewness": data.skew(),
    "kurtosis": data.kurtosis()
}
hook(stats_dict, "descriptive statistics", name='stats', depends_on=['target_col'])
for k, v in stats_dict.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

submit({
    "column": target_col,
    "count": int(stats_dict["count"]),
    "mean": round(float(stats_dict["mean"]), 2),
    "std": round(float(stats_dict["std"]), 2),
    "min": round(float(stats_dict["min"]), 2),
    "max": round(float(stats_dict["max"]), 2),
    "median": round(float(stats_dict["median"]), 2),
    "skewness": round(float(stats_dict["skewness"]), 2),
    "kurtosis": round(float(stats_dict["kurtosis"]), 2)
})
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 1,
    n_steps=3,
    difficulty="EASY",
    alternative_code_templates=[
        # CV-based alternative
        """
# Step 1: Find column with highest coefficient of variation
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
means = df[numeric_cols].mean()
stds = df[numeric_cols].std()
cv = stds / means.abs()
cv = cv.replace([float('inf'), float('-inf')], float('nan')).dropna()
target_col = cv.idxmax()
hook(target_col, "target column (highest CV)", name='target_col')
print(f"Summarizing: {target_col}")

# Step 2: Compute all descriptive statistics
data = df[target_col].dropna()
stats_dict = {
    "count": len(data),
    "mean": data.mean(),
    "std": data.std(),
    "min": data.min(),
    "max": data.max(),
    "median": data.median(),
    "skewness": data.skew(),
    "kurtosis": data.kurtosis()
}
hook(stats_dict, "descriptive statistics", name='stats', depends_on=['target_col'])
for k, v in stats_dict.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

submit({
    "column": target_col,
    "count": int(stats_dict["count"]),
    "mean": round(float(stats_dict["mean"]), 2),
    "std": round(float(stats_dict["std"]), 2),
    "min": round(float(stats_dict["min"]), 2),
    "max": round(float(stats_dict["max"]), 2),
    "median": round(float(stats_dict["median"]), 2),
    "skewness": round(float(stats_dict["skewness"]), 2),
    "kurtosis": round(float(stats_dict["kurtosis"]), 2)
})
""".strip(),
    ],
)

LEVENE_VARIANCE_TEST = CompositionTemplate(
    name="levene_variance_test",
    description="Test equality of variances using Levene's test (first categorical column with 2-6 groups) on highest-variance numeric column (excluding identifier-like columns)",
    output_schema='A JSON object with exactly 7 keys: "target_column", "grouping_column", "n_groups" (integer), "test_statistic" (rounded to 2 decimals), "significance_score" (rounded to 4 decimals), "variances_equal" (boolean, true if significance_score >= 0.05), and "group_variances" (dict mapping group names to variance rounded to 2 decimals). Example: {"target_column": "score", "grouping_column": "class", "n_groups": 3, "test_statistic": 2.34, "significance_score": 0.0987, "variances_equal": true, "group_variances": {"A": 123.4567, "B": 145.6789, "C": 112.3456}}',
    code_template="""
# Step 1: Find numeric column with high variance
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
variances = df[numeric_cols].var()
target_col = variances.idxmax()
hook(target_col, "target column", name='target_col')
print(f"Testing variance homogeneity for: {target_col}")

# Step 2: Find categorical grouping column
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
group_col = None
for col in cat_cols:
    if 2 <= df[col].nunique() <= 6:
        group_col = col
        break

if group_col is None:
    submit({"error": "No suitable categorical column (2-6 groups)"})
else:
    hook(group_col, "grouping column", name='group_col', depends_on=['target_col'])
    print(f"Grouping by: {group_col}")

    # Step 3: Get groups
    groups = df.groupby(group_col)[target_col].apply(lambda x: x.dropna().values)
    group_names = groups.index.tolist()
    n_groups = len(group_names)
    hook({"n_groups": n_groups, "groups": group_names}, "group info", name='group_info', depends_on=['group_col'])

    # Step 4: Compute group variances
    group_vars = {str(g): float(df[df[group_col] == g][target_col].var()) for g in group_names}
    hook(group_vars, "group variances", name='group_vars', depends_on=['group_info'])
    print("Variances by group:")
    for g, v in group_vars.items():
        print(f"  {g}: {v:.4f}")

    # Step 5: Levene's test
    stat, p_value = scipy.stats.levene(*[groups[g] for g in group_names])
    hook({"stat": float(stat), "p_value": float(p_value)}, "Levene test results", name='levene', depends_on=['group_vars'])
    print(f"Levene's test: W={stat:.4f}, p={p_value:.6f}")

    # Step 6: Interpret
    variances_equal = p_value >= 0.05
    hook(variances_equal, "variances equal at alpha=0.05", name='result', depends_on=['levene'])

    submit({
        "target_column": target_col,
        "grouping_column": group_col,
        "n_groups": n_groups,
        "test_statistic": round(float(stat), 2),
        "significance_score": round(float(p_value), 4),
        "variances_equal": variances_equal,
        "group_variances": {k: round(v, 2) for k, v in group_vars.items()}
    })
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 1 and _has_categorical_cols(p),
    n_steps=7,
    difficulty="HARD",
)

KOLMOGOROV_SMIRNOV_NORMALITY = CompositionTemplate(
    name="ks_normality_test",
    description="Test for normality using Kolmogorov-Smirnov test against theoretical normal on column with lowest absolute skewness (closest to zero) (excluding identifier-like columns), alpha=0.05",
    output_schema='A JSON object with exactly 6 keys: "column" (tested column), "test_statistic" (rounded to 2 decimals), "significance_score" (rounded to 4 decimals), "is_normal" (boolean, true if significance_score >= 0.05), "sample_mean" (rounded to 2 decimals), and "sample_std" (rounded to 2 decimals). Example: {"column": "height", "test_statistic": 0.04, "significance_score": 0.2345, "is_normal": true, "sample_mean": 170.1234, "sample_std": 10.5678}',
    code_template="""
# Step 1: Find column closest to normal (lowest skewness)
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
skewness = {col: abs(df[col].skew()) for col in numeric_cols}
target_col = min(skewness, key=skewness.get)
hook(target_col, "target column (lowest skewness)", name='target_col')
print(f"Testing: {target_col} (skewness: {skewness[target_col]:.4f})")

# Step 2: Get clean data
data = df[target_col].dropna()
n = len(data)
hook(n, "sample size", name='n', depends_on=['target_col'])

# Step 3: Compute sample statistics
sample_mean = data.mean()
sample_std = data.std()
hook({"mean": float(sample_mean), "std": float(sample_std)}, "sample statistics", name='stats', depends_on=['n'])
print(f"Mean: {sample_mean:.4f}, Std: {sample_std:.4f}")

# Step 4: Standardize data (handle zero std edge case)
if sample_std == 0:
    submit({"error": "Cannot test normality: all values are identical (std=0)"})
else:
    standardized = (data - sample_mean) / sample_std

    # Step 5: K-S test against standard normal
    ks_stat, p_value = scipy.stats.kstest(standardized, 'norm')
    hook({"ks_stat": float(ks_stat), "p_value": float(p_value)}, "K-S test results", name='ks_results', depends_on=['stats'])
    print(f"K-S statistic: {ks_stat:.4f}, p-value: {p_value:.6f}")

    # Step 6: Interpret
    is_normal = p_value >= 0.05
    hook(is_normal, "is normal at alpha=0.05", name='is_normal', depends_on=['ks_results'])

    submit({
        "column": target_col,
        "test_statistic": round(float(ks_stat), 2),
        "significance_score": round(float(p_value), 4),
        "is_normal": is_normal,
        "sample_mean": round(float(sample_mean), 2),
        "sample_std": round(float(sample_std), 2)
    })
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 1,
    n_steps=7,
    difficulty="MEDIUM",
)

PAIRED_COLUMNS_CORRELATION_CHANGE = CompositionTemplate(
    name="correlation_change_analysis",
    description="Select two most skewed positive numeric columns (excluding identifier-like columns), compare Pearson correlation before and after log1p transformation",
    output_schema='A JSON object with exactly 5 keys: "columns" (list of 2 column names, alphabetically sorted), "original_association" (rounded to 2 decimals), "log_association" (rounded to 2 decimals), "improvement" (absolute difference, rounded to 2 decimals), and "transformation_helpful" (boolean, true if |log_association| > |original_association|). Example: {"columns": ["income", "spending"], "original_association": 0.4567, "log_association": 0.7234, "improvement": 0.2667, "transformation_helpful": true}',
    code_template="""
# Step 1: Find two most skewed positive columns
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
# Filter to positive columns (can take log)
positive_cols = [c for c in numeric_cols if (df[c] > 0).all()]
if len(positive_cols) < 2:
    # Use first two numeric cols - log1p with min-shift handles non-positive values
    positive_cols = numeric_cols[:2]

skewness = {col: abs(df[col].skew()) for col in positive_cols}
# Sort by skewness desc, then alphabetically for deterministic tie-breaking
sorted_cols = sorted(skewness.items(), key=lambda x: (-x[1], x[0]))[:2]
sorted_cols = [col for col, _ in sorted_cols]
col1, col2 = sorted_cols[0], sorted_cols[1]
hook([col1, col2], "selected columns (most skewed)", name='selected_cols')
print(f"Analyzing: {col1} vs {col2}")

# Step 2: Compute original Pearson correlation
clean_data = df[[col1, col2]].dropna()
orig_corr = clean_data[col1].corr(clean_data[col2])
hook(orig_corr, "original correlation", name='orig_corr', depends_on=['selected_cols'])
print(f"Original Pearson: {orig_corr:.4f}")

# Step 3: Log-transform (add small constant to avoid log(0))
log_col1 = np.log1p(clean_data[col1] - clean_data[col1].min() + 1)
log_col2 = np.log1p(clean_data[col2] - clean_data[col2].min() + 1)
hook({"col1_skew_after": log_col1.skew(), "col2_skew_after": log_col2.skew()}, "skewness after transform", name='log_skew', depends_on=['orig_corr'])

# Step 4: Compute log-transformed correlation
log_corr = log_col1.corr(log_col2)
hook(log_corr, "log-transformed correlation", name='log_corr', depends_on=['log_skew'])
print(f"Log-transformed Pearson: {log_corr:.4f}")

# Step 5: Compare
improvement = abs(log_corr) - abs(orig_corr)
transformation_helpful = abs(log_corr) > abs(orig_corr)
hook({"improvement": float(improvement), "helpful": transformation_helpful}, "comparison", name='comparison', depends_on=['log_corr'])

submit({
    "columns": sorted([col1, col2]),
    "original_association": round(float(orig_corr), 2),
    "log_association": round(float(log_corr), 2),
    "improvement": round(abs(float(improvement)), 2),
    "transformation_helpful": transformation_helpful
})
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 2,
    n_steps=6,
    difficulty="HARD",
)


# =============================================================================
# ADVANCED TEMPLATES - Audit/Fix/Recompute, Multi-hop, Adaptive, Iteration
# =============================================================================

AUDIT_MISSING_IMPUTATION_EFFECT = CompositionTemplate(
    name="audit_missing_imputation_effect",
    description="Find column with most missing values (excluding identifier-like columns), impute with median, measure mean shift",
    output_schema='A JSON object with exactly 6 keys: "column" (name of numeric column with highest missing_pct), "missing_pct" (rounded to 2 decimals), "imputed_count" (integer), "original_mean" (rounded to 2 decimals, computed on non-missing), "imputed_mean" (rounded to 2 decimals, mean after median imputation), and "mean_shift" (imputed_mean - original_mean, rounded to 2 decimals). Example: {"column": "age", "missing_pct": 12.50, "imputed_count": 125, "original_mean": 35.4567, "imputed_mean": 35.1023, "mean_shift": -0.3544}',
    code_template="""
# Step 1: Find numeric column with highest missing percentage
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
missing_pcts = {col: df[col].isna().mean() * 100 for col in numeric_cols}
# Alphabetical tiebreaker for determinism
target_col = sorted(missing_pcts.items(), key=lambda x: (-x[1], x[0]))[0][0]
missing_pct = missing_pcts[target_col]
hook(target_col, "column with most missing", name='target_col')
print(f"Target column: {target_col} ({missing_pct:.2f}% missing)")

if missing_pct == 0:
    submit({"error": "No missing values in any numeric column"})
else:
    # Step 2: Compute original mean (on non-missing values)
    original_mean = df[target_col].dropna().mean()
    hook(original_mean, "original mean (non-missing)", name='original_mean', depends_on=['target_col'])
    print(f"Original mean: {original_mean:.4f}")

    # Step 3: Impute with median
    median_val = df[target_col].median()
    imputed_count = df[target_col].isna().sum()
    filled = df[target_col].fillna(median_val)
    hook({"median": float(median_val), "imputed_count": int(imputed_count)}, "imputation info", name='imputation', depends_on=['original_mean'])
    print(f"Imputing {imputed_count} values with median {median_val:.4f}")

    # Step 4: Compute imputed mean
    imputed_mean = filled.mean()
    hook(imputed_mean, "mean after imputation", name='imputed_mean', depends_on=['imputation'])
    print(f"Imputed mean: {imputed_mean:.4f}")

    # Step 5: Compute shift
    mean_shift = imputed_mean - original_mean
    hook(mean_shift, "mean shift", name='mean_shift', depends_on=['imputed_mean'])
    print(f"Mean shift: {mean_shift:+.4f}")

    submit({
        "column": target_col,
        "missing_pct": round(missing_pct, 2),
        "imputed_count": int(imputed_count),
        "original_mean": round(float(original_mean), 2),
        "imputed_mean": round(float(imputed_mean), 2),
        "mean_shift": round(float(mean_shift), 2)
    })
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _has_missing_numeric(p),
    n_steps=6,
    difficulty="HARD",
)

RANK_CATEGORIES_BY_TARGET_MEAN = CompositionTemplate(
    name="rank_categories_by_target_mean",
    description="Use first categorical column with 3-20 unique values as grouping, highest-variance numeric column as target (excluding identifier-like columns), rank top 3 categories by mean (only categories with n >= 10 samples)",
    output_schema='A JSON object with exactly 3 keys: "category_column" (string), "target_column" (string), and "ranking" (list of top category objects, each with "category", "mean" rounded to 2 decimals, and "n", sorted by mean descending). Example: {"category_column": "region", "target_column": "sales", "ranking": [{"category": "West", "mean": 1234.5678, "n": 42}, {"category": "North", "mean": 1150.0000, "n": 31}]}',
    code_template="""
# Step 1: Find suitable categorical column (moderate cardinality)
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
suitable_cat = None
for col in cat_cols:
    n_unique = df[col].nunique()
    if 3 <= n_unique <= 20:
        suitable_cat = col
        break

if suitable_cat is None:
    submit({"error": "No categorical column with 3-20 unique values found"})
else:
    hook(suitable_cat, "category column", name='cat_col')
    print(f"Category column: {suitable_cat} ({df[suitable_cat].nunique()} categories)")

    # Step 2: Find target column (highest variance numeric)
    numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
    variances = df[numeric_cols].var()
    target_col = variances.idxmax()
    hook(target_col, "target column", name='target_col', depends_on=['cat_col'])
    print(f"Target column: {target_col}")

    # Step 3: Compute group statistics
    min_n = {min_n}
    top_k = {top_k}
    grouped = df.groupby(suitable_cat)[target_col].agg(['mean', 'count'])
    hook(len(grouped), "total categories", name='n_cats', depends_on=['target_col'])

    # Step 4: Filter by minimum sample size
    filtered = grouped[grouped['count'] >= min_n]
    hook(len(filtered), "categories meeting min_n", name='n_filtered', depends_on=['n_cats'])
    print(f"Categories with n >= {min_n}: {len(filtered)}/{len(grouped)}")

    # Step 5: Rank and take top_k
    ranked = filtered.sort_values('mean', ascending=False).head(top_k)
    ranking = [
        {"category": str(idx), "mean": round(row['mean'], 2), "n": int(row['count'])}
        for idx, row in ranked.iterrows()
    ]
    hook(ranking, "final ranking", name='ranking', depends_on=['n_filtered'])

    submit({
        "category_column": suitable_cat,
        "target_column": target_col,
        "ranking": ranking
    })
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 1 and _has_categorical_cols(p),
    n_steps=6,
    difficulty="HARD",
    param_sets=[
        # Use consistent params for triangulation
        {"min_n": 10, "top_k": 3},
    ],
)

ADAPTIVE_TWO_SAMPLE_TEST = CompositionTemplate(
    name="adaptive_two_sample_test",
    description="Select least-skewed numeric column as target and first binary categorical as grouping, choose t-test or Mann-Whitney based on normality (Shapiro-Wilk) and variance (Levene) assumptions",
    output_schema='A JSON object with exactly 10 keys: "target_column", "grouping_column", "group1", "group2", "method_choice" ("parametric" or "rank_based"), "significance_score" (rounded to 4 decimals), "effect_size_kind" ("standardized_diff" or "rank_based"), "effect_size" (rounded to 2 decimals), "significant" (boolean), and "assumptions_passed" (boolean). Example: {"target_column": "score", "grouping_column": "treatment", "group1": "control", "group2": "drug", "method_choice": "rank_based", "significance_score": 0.0345, "effect_size_kind": "rank_based", "effect_size": 0.2451, "significant": true, "assumptions_passed": false}',
    code_template="""
# Step 1: Find target column (prefer less skewed for interesting test choice)
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
skewness = {col: abs(df[col].skew()) for col in numeric_cols}
target_col = min(skewness, key=skewness.get)
hook(target_col, "target column (least skewed)", name='target_col')
print(f"Target: {target_col} (skewness: {skewness[target_col]:.4f})")

# Step 2: Find binary grouping column
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
binary_col = None
for col in cat_cols:
    if df[col].nunique() == 2:
        binary_col = col
        break

if binary_col is None:
    submit({"error": "No binary categorical column found"})
else:
    hook(binary_col, "grouping column", name='group_col', depends_on=['target_col'])
    groups = df[binary_col].dropna().unique()
    g1_name, g2_name = str(groups[0]), str(groups[1])
    print(f"Groups: {g1_name} vs {g2_name}")

    # Step 3: Extract group data
    g1_data = df[df[binary_col] == groups[0]][target_col].dropna()
    g2_data = df[df[binary_col] == groups[1]][target_col].dropna()
    hook({"n1": len(g1_data), "n2": len(g2_data)}, "group sizes", name='sizes', depends_on=['group_col'])

    # Step 4: Check normality (Shapiro-Wilk, sample if large)
    sample1 = g1_data.sample(min(5000, len(g1_data)), random_state=42) if len(g1_data) > 5000 else g1_data
    sample2 = g2_data.sample(min(5000, len(g2_data)), random_state=42) if len(g2_data) > 5000 else g2_data
    _, p_norm1 = scipy.stats.shapiro(sample1)
    _, p_norm2 = scipy.stats.shapiro(sample2)
    normal_ok = (p_norm1 >= 0.05) and (p_norm2 >= 0.05)
    hook({"p_norm1": p_norm1, "p_norm2": p_norm2, "normal_ok": normal_ok}, "normality check", name='normality', depends_on=['sizes'])

    # Step 5: Check equal variance (Levene)
    _, p_levene = scipy.stats.levene(g1_data, g2_data)
    variance_ok = p_levene >= 0.05
    hook({"p_levene": p_levene, "variance_ok": variance_ok}, "variance check", name='variance', depends_on=['normality'])

    assumptions_passed = normal_ok and variance_ok
    hook(assumptions_passed, "assumptions passed", name='assumptions', depends_on=['variance'])
    print(f"Assumptions passed: {assumptions_passed}")

    # Step 6: Choose and run appropriate test
    if assumptions_passed:
        test_used = "parametric"
        t_stat, p_value = scipy.stats.ttest_ind(g1_data, g2_data)
        pooled_std = np.sqrt(((len(g1_data)-1)*g1_data.std()**2 + (len(g2_data)-1)*g2_data.std()**2) / (len(g1_data)+len(g2_data)-2))
        effect_size = (g1_data.mean() - g2_data.mean()) / pooled_std if pooled_std > 0 else 0
        effect_size_type = "standardized_diff"
    else:
        test_used = "rank_based"
        u_stat, p_value = scipy.stats.mannwhitneyu(g1_data, g2_data, alternative='two-sided')
        n1, n2 = len(g1_data), len(g2_data)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)
        effect_size_type = "rank_based"

    hook({"test": test_used, "p_value": p_value, "effect_size": effect_size}, "test results", name='results', depends_on=['assumptions'])
    print(f"Test: {test_used}, p={p_value:.6f}, effect={effect_size:.4f}")

    submit({
        "target_column": target_col,
        "grouping_column": binary_col,
        "group1": g1_name,
        "group2": g2_name,
        "method_choice": test_used,
        "significance_score": round(float(p_value), 4),
        "effect_size_kind": effect_size_type,
        "effect_size": round(float(effect_size), 2),
        "significant": p_value < 0.05,
        "assumptions_passed": assumptions_passed
    })
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 1 and _has_categorical_cols(p),
    n_steps=9,
    difficulty="VERY_HARD",
)

QUANTILE_BIN_BEST_MEAN = CompositionTemplate(
    name="quantile_bin_best_mean",
    description="Select most-skewed numeric column as binning column, highest-variance other column as target (excluding identifier-like columns), bin into quantiles, find which bin has highest mean of target",
    output_schema='A JSON object with exactly 5 keys: "binning_column" (string), "target_column" (string), "best_bin" (string label), "best_bin_mean" (rounded to 2 decimals), and "bin_means" (dict mapping bin labels to means rounded to 2 decimals). Example: {"binning_column": "age", "target_column": "income", "best_bin": "(45.0, 60.0]", "best_bin_mean": 75000.1234, "bin_means": {"(18.0, 30.0]": 45000.0, "(30.0, 45.0]": 52000.0}}',
    code_template="""
# Step 1: Choose binning column (most skewed - more interesting bins)
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
if len(numeric_cols) < 2:
    submit({"error": "Need at least 2 numeric columns"})
else:
    skewness = {col: abs(df[col].skew()) for col in numeric_cols}
    # Alphabetical tiebreaker for determinism
    bin_col = sorted(skewness.items(), key=lambda x: (-x[1], x[0]))[0][0]
    hook(bin_col, "binning column (most skewed)", name='bin_col')
    print(f"Binning column: {bin_col} (skewness: {skewness[bin_col]:.4f})")

    # Step 2: Choose target column (highest variance, different from bin_col)
    other_cols = [c for c in numeric_cols if c != bin_col]
    variances = df[other_cols].var()
    target_col = variances.idxmax()
    hook(target_col, "target column", name='target_col', depends_on=['bin_col'])
    print(f"Target column: {target_col}")

    # Step 3: Create quantile bins
    n_bins = {n_bins}
    try:
        bins = pd.qcut(df[bin_col], q=n_bins, duplicates='drop')
        hook(bins.nunique(), "actual bin count", name='bin_count', depends_on=['target_col'])
    except ValueError:
        submit({"error": "Cannot create quantile bins - too few unique values"})
    else:
        # Step 4: Compute mean per bin
        bin_means = df.groupby(bins)[target_col].mean()
        bin_means_dict = {str(k): round(v, 2) for k, v in bin_means.items()}
        hook(bin_means_dict, "means per bin", name='bin_means', depends_on=['bin_count'])
        print("Bin means:")
        for b, m in sorted(bin_means.items(), key=lambda x: x[1], reverse=True):
            print(f"  {b}: {m:.4f}")

        # Step 5: Find best bin
        best_bin = bin_means.idxmax()
        best_mean = bin_means.max()
        hook({"best_bin": str(best_bin), "mean": best_mean}, "best bin", name='best', depends_on=['bin_means'])

        submit({
            "binning_column": bin_col,
            "target_column": target_col,
            "best_bin": str(best_bin),
            "best_bin_mean": round(float(best_mean), 2),
            "bin_means": bin_means_dict
        })
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 2,
    n_steps=7,
    difficulty="HARD",
    param_sets=[
        # Use consistent params for triangulation
        {"n_bins": 4},
    ],
)

ITERATIVE_OUTLIER_REMOVAL = CompositionTemplate(
    name="iterative_outlier_removal",
    description="Iteratively remove z-score outliers (|z|>3) from highest-CV column (excluding identifier-like columns) until no outliers remain OR mean change < 0.001, report convergence",
    output_schema='A JSON object with exactly 5 keys: "column" (string), "iterations_used" (integer), "total_removed" (integer), "initial_mean" (rounded to 2 decimals), and "final_mean" (rounded to 2 decimals). Example: {"column": "price", "iterations_used": 3, "total_removed": 18, "initial_mean": 123.4567, "final_mean": 118.9012}',
    code_template="""
# Step 1: Pick column with highest CV (most likely to have meaningful outliers)
numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
cv_values = {}
for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()
    if mean != 0 and std > 0:
        cv_values[col] = std / abs(mean)

if not cv_values:
    submit({"error": "No suitable numeric columns found"})
else:
    # Alphabetical tiebreaker for determinism
    target_col = sorted(cv_values.items(), key=lambda x: (-x[1], x[0]))[0][0]
    hook(target_col, "target column (highest CV)", name='target_col')
    print(f"Target: {target_col} (CV: {cv_values[target_col]:.4f})")

    # Step 2: Initialize
    z_threshold = {z_threshold}
    max_iter = {max_iter}
    eps = {eps}
    
    data = df[target_col].dropna().copy()
    initial_mean = data.mean()
    hook({"n": len(data), "mean": initial_mean}, "initial state", name='initial', depends_on=['target_col'])
    print(f"Initial: n={len(data)}, mean={initial_mean:.4f}")

    # Step 3: Iterative removal
    total_removed = 0
    prev_mean = initial_mean
    iterations_used = 0

    for i in range(max_iter):
        mean = data.mean()
        std = data.std()
        if std == 0:
            break
        
        mask = (data - mean).abs() <= z_threshold * std
        removed_this_iter = (~mask).sum()
        
        if removed_this_iter == 0:
            iterations_used = i + 1
            break
            
        data = data[mask]
        total_removed += removed_this_iter
        new_mean = data.mean()
        
        hook({"iter": i+1, "removed": removed_this_iter, "mean": new_mean}, f"iteration {i+1}", name=f'iter_{i+1}', depends_on=['initial'])
        print(f"  Iter {i+1}: removed {removed_this_iter}, mean={new_mean:.4f}")
        
        if abs(new_mean - prev_mean) < eps:
            iterations_used = i + 1
            break
        prev_mean = new_mean
        iterations_used = i + 1

    final_mean = data.mean()
    hook({"iterations": iterations_used, "total_removed": total_removed, "final_mean": final_mean}, "convergence", name='convergence', depends_on=['initial'])
    print(f"Converged: {iterations_used} iterations, {total_removed} removed")

    submit({
        "column": target_col,
        "iterations_used": iterations_used,
        "total_removed": int(total_removed),
        "initial_mean": round(float(initial_mean), 2),
        "final_mean": round(float(final_mean), 2)
    })
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 1,
    n_steps=9,
    difficulty="VERY_HARD",
    param_sets=[
        # Use consistent params - unhinted model may choose different values
        # but computed outputs should match given the same algorithm
        {"z_threshold": 3.0, "max_iter": 10, "eps": 0.001},
    ],
)

SEGMENT_THEN_ANALYZE_OUTLIERS = CompositionTemplate(
    name="segment_then_analyze_outliers",
    description="Use first categorical column with 3-15 unique values as grouping, highest-variance numeric column as target (excluding identifier-like columns), find category with highest mean, analyze IQR-based outlier rate within that segment",
    output_schema='A JSON object with exactly 7 keys: "category_column" (string), "target_column" (string), "best_category" (string with highest mean), "segment_size" (integer), "segment_mean" (rounded to 2 decimals), "segment_outliers" (integer, IQR method), and "outlier_rate" (rounded to 2 decimals). Example: {"category_column": "region", "target_column": "sales", "best_category": "West", "segment_size": 120, "segment_mean": 1234.5678, "segment_outliers": 9, "outlier_rate": 0.0750}',
    code_template="""
# Step 1: Find suitable categorical column
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
cat_col = None
for col in cat_cols:
    n_unique = df[col].nunique()
    if 3 <= n_unique <= 15:
        cat_col = col
        break

if cat_col is None:
    submit({"error": "No suitable categorical column (3-15 categories)"})
else:
    hook(cat_col, "category column", name='cat_col')
    print(f"Category column: {cat_col}")

    # Step 2: Find target column (highest variance)
    numeric_cols = [c for c in df.select_dtypes('number').columns.tolist()
                if not _is_index_column(c)]
    variances = df[numeric_cols].var()
    target_col = variances.idxmax()
    hook(target_col, "target column", name='target_col', depends_on=['cat_col'])
    print(f"Target column: {target_col}")

    # Step 3: Find best category (highest mean)
    means = df.groupby(cat_col)[target_col].mean()
    best_cat = means.idxmax()
    hook({"category": str(best_cat), "mean": means[best_cat]}, "best category", name='best_cat', depends_on=['target_col'])
    print(f"Best category: {best_cat} (mean: {means[best_cat]:.4f})")

    # Step 4: Extract segment
    segment = df[df[cat_col] == best_cat][target_col].dropna()
    segment_size = len(segment)
    segment_mean = segment.mean()
    hook({"size": segment_size, "mean": segment_mean}, "segment stats", name='segment', depends_on=['best_cat'])
    print(f"Segment size: {segment_size}")

    # Step 5: Compute IQR outliers within segment
    q1 = segment.quantile(0.25)
    q3 = segment.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((segment < lower) | (segment > upper)).sum()
    outlier_rate = outliers / segment_size if segment_size > 0 else 0
    hook({"outliers": int(outliers), "rate": outlier_rate}, "outlier analysis", name='outliers', depends_on=['segment'])
    print(f"Outliers: {outliers} ({outlier_rate:.2%})")

    submit({
        "category_column": cat_col,
        "target_column": target_col,
        "best_category": str(best_cat),
        "segment_size": segment_size,
        "segment_mean": round(float(segment_mean), 2),
        "segment_outliers": int(outliers),
        "outlier_rate": round(float(outlier_rate), 2)
    })
""".strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 1 and _has_categorical_cols(p),
    n_steps=7,
    difficulty="HARD",
)


# =============================================================================
# ALL TEMPLATES REGISTRY
# =============================================================================

ALL_TEMPLATES = [
    # === EASY ===
    COUNT_HIGH_MISSING_COLUMNS,
    DUPLICATE_ROWS_SUMMARY,
    IDENTIFIER_LIKE_COLUMNS,
    COEFFICIENT_OF_VARIATION,
    IQR_OUTLIER_ANALYSIS,
    PERCENTILE_RANKING,
    DESCRIPTIVE_SUMMARY,
    # === MEDIUM ===
    MAX_VARIANCE_MEAN,
    MIN_MEAN_COLUMN_STD,
    STRONGEST_CORRELATION,
    WEAKEST_CORRELATION,
    COUNT_OUTLIER_COLUMNS,
    CHI_SQUARED_INDEPENDENCE,
    MANN_WHITNEY_U_TEST,
    SPEARMAN_RANK_CORRELATION,
    KOLMOGOROV_SMIRNOV_NORMALITY,
    # === HARD ===
    CONDITIONAL_NORMALITY,
    CATEGORY_WITH_HIGHEST_TARGET_MEAN,
    CORRELATION_AFTER_OUTLIER_REMOVAL,
    LEVENE_VARIANCE_TEST,
    PAIRED_COLUMNS_CORRELATION_CHANGE,
    REGRESSION_MOST_PREDICTIVE,
    TTEST_DISCOVERED_GROUPS,
    AUDIT_MISSING_IMPUTATION_EFFECT,
    RANK_CATEGORIES_BY_TARGET_MEAN,
    QUANTILE_BIN_BEST_MEAN,
    SEGMENT_THEN_ANALYZE_OUTLIERS,
    # === VERY_HARD ===
    BOOTSTRAP_CI_DISCOVERED,
    ANOVA_DISCOVERED_GROUPS,
    MULTIPLE_REGRESSION_TOP_PREDICTORS,
    ADAPTIVE_TWO_SAMPLE_TEST,
    ITERATIVE_OUTLIER_REMOVAL,
]


def get_applicable_templates(profile: dict) -> list[CompositionTemplate]:
    """Return templates that are applicable to the given dataset profile.

    Returns templates sorted by n_steps descending (hardest first).
    """
    applicable = [t for t in ALL_TEMPLATES if t.is_applicable(profile)]
    return sorted(applicable, key=lambda t: t.n_steps, reverse=True)

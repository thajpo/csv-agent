"""
Composition templates for deterministic question generation.

Each template defines a code pattern that:
1. Discovers something about the data (forces exploration)
2. Computes a verifiable result
3. Can be verbalized into a natural language question
"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class CompositionTemplate:
    """A template for generating a compositional question."""

    name: str
    description: str  # Human-readable description of what this computes
    code_template: str  # Python code with {placeholders}
    output_type: str  # "scalar", "dict", "list"
    applicable_when: Callable[[dict], bool]  # Profile -> bool
    n_steps: int
    difficulty: str  # EASY, MEDIUM, HARD, VERY_HARD

    def is_applicable(self, profile: dict) -> bool:
        """Check if this template can be used with the given dataset profile."""
        try:
            return self.applicable_when(profile)
        except Exception:
            return False

    def instantiate(self, profile: dict) -> str:
        """Fill in template placeholders with profile data."""
        # Get numeric columns
        numeric_cols = [
            col for col, info in profile.get("columns", {}).items()
            if info.get("type") == "numeric"
        ]

        # Get categorical columns
        categorical_cols = [
            col for col, info in profile.get("columns", {}).items()
            if info.get("type") == "categorical"
        ]

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

        return code


def _count_numeric_cols(profile: dict) -> int:
    """Count numeric columns in profile."""
    return len([
        c for c in profile.get("columns", {}).values()
        if c.get("type") == "numeric"
    ])


def _has_missing_values(profile: dict) -> bool:
    """Check if any column has missing values."""
    return any(
        c.get("missing_pct", 0) > 0
        for c in profile.get("columns", {}).values()
    )


def _has_categorical_cols(profile: dict) -> bool:
    """Check if dataset has categorical columns."""
    return any(
        c.get("type") == "categorical"
        for c in profile.get("columns", {}).values()
    )


# =============================================================================
# SUPERLATIVE TEMPLATES - Find column with property X, compute Y on it
# =============================================================================

MAX_VARIANCE_MEAN = CompositionTemplate(
    name="max_variance_mean",
    description="Find the numeric column with highest variance, then compute its mean",
    code_template='''
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
'''.strip(),
    output_type="scalar",
    applicable_when=lambda p: _count_numeric_cols(p) >= 2,
    n_steps=3,
    difficulty="MEDIUM",
)

MIN_MEAN_COLUMN_STD = CompositionTemplate(
    name="min_mean_column_std",
    description="Find the numeric column with lowest mean, then compute its standard deviation",
    code_template='''
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
'''.strip(),
    output_type="scalar",
    applicable_when=lambda p: _count_numeric_cols(p) >= 2,
    n_steps=3,
    difficulty="MEDIUM",
)


# =============================================================================
# CROSS-COLUMN TEMPLATES - Discover relationships between columns
# =============================================================================

STRONGEST_CORRELATION = CompositionTemplate(
    name="strongest_correlation",
    description="Find the pair of numeric columns with the strongest correlation",
    code_template='''
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

submit({"columns": list(max_corr_idx), "correlation": round(float(correlation_value), 3)})
'''.strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 3,
    n_steps=4,
    difficulty="MEDIUM",
)

WEAKEST_CORRELATION = CompositionTemplate(
    name="weakest_correlation",
    description="Find the pair of numeric columns with the weakest (closest to zero) correlation",
    code_template='''
# Step 1: Compute correlation matrix
numeric_cols = df.select_dtypes('number')
corr_matrix = numeric_cols.corr().abs()
hook(corr_matrix.shape, "correlation matrix computed", name='corr_shape')

# Step 2: Find minimum non-zero correlation (mask diagonal and zeros)
np.fill_diagonal(corr_matrix.values, np.nan)
min_corr_idx = corr_matrix.stack().idxmin()
hook(min_corr_idx, "min_corr_idx found", name='min_corr_idx')
print(f"Weakest correlation pair: {min_corr_idx}")

# Step 3: Get the correlation value
correlation_value = corr_matrix.loc[min_corr_idx[0], min_corr_idx[1]]
hook(correlation_value, "correlation_value extracted", name='correlation_value', depends_on=['min_corr_idx'])
print(f"Correlation: {correlation_value:.4f}")

submit({"columns": list(min_corr_idx), "correlation": round(float(correlation_value), 3)})
'''.strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 3,
    n_steps=4,
    difficulty="MEDIUM",
)


# =============================================================================
# CONDITIONAL TEMPLATES - Test condition, branch based on result
# =============================================================================

CONDITIONAL_NORMALITY = CompositionTemplate(
    name="conditional_normality",
    description="Test if column is normally distributed; report mean±std if yes, median+IQR if no",
    code_template='''
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
'''.strip(),
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
    code_template='''
# Step 1: Calculate missing percentage for each column
missing_pct = df.isnull().mean() * 100
hook(missing_pct.to_dict(), "missing percentages computed", name='missing_pct')
print("Missing percentages:")
print(missing_pct.sort_values(ascending=False).head(10))

# Step 2: Count columns above threshold
threshold = 5.0
high_missing_cols = (missing_pct > threshold).sum()
hook(high_missing_cols, f"count of columns with >{threshold}% missing", name='high_missing_count', depends_on=['missing_pct'])
print(f"Columns with >{threshold}% missing: {high_missing_cols}")

# Step 3: List which columns
high_missing_names = missing_pct[missing_pct > threshold].index.tolist()
hook(high_missing_names, "names of high-missing columns", name='high_missing_names', depends_on=['missing_pct'])

submit({"count": int(high_missing_cols), "columns": high_missing_names})
'''.strip(),
    output_type="dict",
    applicable_when=lambda p: True,  # Always applicable
    n_steps=3,
    difficulty="EASY",
)

COUNT_OUTLIER_COLUMNS = CompositionTemplate(
    name="count_outlier_columns",
    description="Count how many numeric columns contain outliers (beyond 3 std from mean)",
    code_template='''
# Step 1: For each numeric column, check for outliers
numeric_cols = df.select_dtypes('number').columns
outlier_counts = {}

for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()
    if std > 0:
        outliers = ((df[col] - mean).abs() > 3 * std).sum()
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
'''.strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 1,
    n_steps=4,
    difficulty="MEDIUM",
)


# =============================================================================
# MULTI-STAGE TEMPLATES - Discovery + Transform + Aggregate chains
# =============================================================================

CATEGORY_WITH_HIGHEST_TARGET_MEAN = CompositionTemplate(
    name="category_highest_target_mean",
    description="Find which category has the highest mean for the most variable numeric column",
    code_template='''
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
'''.strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 1 and _has_categorical_cols(p),
    n_steps=5,
    difficulty="HARD",
)

CORRELATION_AFTER_OUTLIER_REMOVAL = CompositionTemplate(
    name="correlation_after_outlier_removal",
    description="Find strongest correlation, remove outliers from those columns, recompute correlation",
    code_template='''
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
def remove_outliers(series):
    mean, std = series.mean(), series.std()
    return series[(series - mean).abs() <= 3 * std]

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
    "columns": [col1, col2],
    "original_correlation": round(float(original_corr), 3),
    "outliers_removed": int(rows_removed),
    "clean_correlation": round(float(clean_corr), 3)
})
'''.strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 3,
    n_steps=5,
    difficulty="HARD",
)


# =============================================================================
# ALL TEMPLATES REGISTRY
# =============================================================================

ALL_TEMPLATES = [
    # Superlative (2)
    MAX_VARIANCE_MEAN,
    MIN_MEAN_COLUMN_STD,
    # Cross-column (2)
    STRONGEST_CORRELATION,
    WEAKEST_CORRELATION,
    # Conditional (1)
    CONDITIONAL_NORMALITY,
    # Threshold (2)
    COUNT_HIGH_MISSING_COLUMNS,
    COUNT_OUTLIER_COLUMNS,
    # Multi-stage (2)
    CATEGORY_WITH_HIGHEST_TARGET_MEAN,
    CORRELATION_AFTER_OUTLIER_REMOVAL,
]


def get_applicable_templates(profile: dict) -> list[CompositionTemplate]:
    """Return templates that are applicable to the given dataset profile."""
    return [t for t in ALL_TEMPLATES if t.is_applicable(profile)]

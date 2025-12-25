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
    output_schema: str  # Exact format description for the answer
    applicable_when: Callable[[dict], bool]  # Profile -> bool
    n_steps: int
    difficulty: str  # EASY, MEDIUM, HARD, VERY_HARD
    param_sets: list[dict[str, Any]] | None = None

    def is_applicable(self, profile: dict) -> bool:
        """Check if this template can be used with the given dataset profile."""
        try:
            return self.applicable_when(profile)
        except Exception:
            return False

    def instantiate(self, profile: dict, params: dict[str, Any] | None = None) -> str:
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
        if params:
            for key, value in params.items():
                code = code.replace(f"{{{key}}}", repr(value))

        return code

    def iter_param_sets(self) -> list[dict[str, Any]]:
        """Return all parameter sets for this template."""
        return self.param_sets or [{}]


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
    output_schema="A single number (the mean), rounded to 3 decimal places",
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
    output_schema="A single number (the standard deviation), rounded to 3 decimal places",
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
    output_schema='A JSON object with exactly two keys: "columns" (a list of the two column names, alphabetically sorted) and "correlation" (the correlation coefficient rounded to 3 decimal places). Example: {"columns": ["col_a", "col_b"], "correlation": 0.847}',
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

submit({"columns": sorted(list(max_corr_idx)), "correlation": round(float(correlation_value), 3)})
'''.strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 3,
    n_steps=4,
    difficulty="MEDIUM",
)

WEAKEST_CORRELATION = CompositionTemplate(
    name="weakest_correlation",
    description="Find the pair of numeric columns with the weakest (closest to zero) correlation",
    output_schema='A JSON object with exactly two keys: "columns" (a list of the two column names, alphabetically sorted) and "correlation" (the absolute correlation value rounded to 3 decimal places). Example: {"columns": ["col_a", "col_b"], "correlation": 0.012}',
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

submit({"columns": sorted(list(min_corr_idx)), "correlation": round(float(correlation_value), 3)})
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
    output_schema='A JSON object with exactly 3 keys. If normal: {"distribution": "normal", "mean": <number>, "std": <number>}. If non-normal: {"distribution": "non-normal", "median": <number>, "iqr": <number>}. The "iqr" value is Q3 minus Q1 as a single number. All numeric values rounded to 3 decimal places.',
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
    output_schema='A JSON object with exactly 2 keys: "count" (integer) and "columns" (list of column names, alphabetically sorted). Example: {"count": 3, "columns": ["col_a", "col_b", "col_c"]}',
    code_template='''
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
'''.strip(),
    output_type="dict",
    applicable_when=lambda p: True,  # Always applicable
    n_steps=3,
    difficulty="EASY",
    param_sets=[
        {"missing_threshold": 5.0},
        {"missing_threshold": 10.0},
    ],
)

COUNT_OUTLIER_COLUMNS = CompositionTemplate(
    name="count_outlier_columns",
    description="Count how many numeric columns contain outliers (beyond 3 std from mean)",
    output_schema='A JSON object with exactly 2 keys: "columns_with_outliers" (integer count of columns containing at least one outlier) and "total_outliers" (integer total count of outlier values across all columns). Example: {"columns_with_outliers": 3, "total_outliers": 47}',
    code_template='''
# Step 1: For each numeric column, check for outliers
numeric_cols = df.select_dtypes('number').columns
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
'''.strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 1,
    n_steps=4,
    difficulty="MEDIUM",
    param_sets=[
        {"z_threshold": 3.0},
        {"z_threshold": 2.5},
    ],
)


# =============================================================================
# MULTI-STAGE TEMPLATES - Discovery + Transform + Aggregate chains
# =============================================================================

CATEGORY_WITH_HIGHEST_TARGET_MEAN = CompositionTemplate(
    name="category_highest_target_mean",
    description="Find which category has the highest mean for the most variable numeric column",
    output_schema='A JSON object with exactly 4 keys: "category_column" (the grouping column name), "best_category" (the category value with highest mean), "target_column" (the numeric column analyzed), and "mean_value" (rounded to 3 decimal places). Example: {"category_column": "region", "best_category": "West", "target_column": "sales", "mean_value": 1234.567}',
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
    output_schema='A JSON object with exactly 4 keys: "columns" (list of 2 column names, alphabetically sorted), "original_correlation" (rounded to 3 decimals), "outliers_removed" (integer count), and "clean_correlation" (rounded to 3 decimals). Example: {"columns": ["col_a", "col_b"], "original_correlation": 0.847, "outliers_removed": 12, "clean_correlation": 0.891}',
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
    "columns": sorted([col1, col2]),
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
# COMPLEX TEMPLATES - Statistical tests & ML (7-10 steps)
# =============================================================================

REGRESSION_MOST_PREDICTIVE = CompositionTemplate(
    name="regression_most_predictive",
    description="Find the most predictive feature via correlation, build regression, report R-squared",
    output_schema='A JSON object with exactly 6 keys: "target" (column name), "best_predictor" (column name), "correlation" (rounded to 3 decimals), "r_squared" (rounded to 4 decimals), "coefficient" (rounded to 4 decimals), and "p_value" (rounded to 6 decimals). Example: {"target": "price", "best_predictor": "sqft", "correlation": 0.834, "r_squared": 0.6956, "coefficient": 135.2847, "p_value": 0.000001}',
    code_template='''
# Step 1: Identify numeric columns
numeric_cols = df.select_dtypes('number').columns.tolist()
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
    "correlation": round(float(best_corr), 3),
    "r_squared": round(float(r_squared), 4),
    "coefficient": round(float(coef), 4),
    "p_value": round(float(p_value), 6)
})
'''.strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 3,
    n_steps=8,
    difficulty="HARD",
)

TTEST_DISCOVERED_GROUPS = CompositionTemplate(
    name="ttest_discovered_groups",
    description="Find binary categorical column, perform t-test on highest-variance numeric column",
    output_schema='A JSON object with exactly 9 keys: "target_column", "grouping_column", "group1", "group2", "mean1" (rounded to 4 decimals), "mean2" (rounded to 4 decimals), "t_statistic" (rounded to 4 decimals), "p_value" (rounded to 6 decimals), and "significant" (boolean, true if p < 0.05). Example: {"target_column": "score", "grouping_column": "gender", "group1": "M", "group2": "F", "mean1": 75.4321, "mean2": 78.1234, "t_statistic": -2.3456, "p_value": 0.019234, "significant": true}',
    code_template='''
# Step 1: Find numeric column with highest variance
numeric_cols = df.select_dtypes('number').columns.tolist()
variances = df[numeric_cols].var()
target_col = variances.idxmax()
hook(target_col, "target column (highest variance)", name='target_col')
print(f"Target: {target_col}")

# Step 2: Find a binary categorical column
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
binary_col = None
for col in cat_cols:
    if df[col].nunique() == 2:
        binary_col = col
        break

if binary_col is None:
    # Create binary from numeric if no categorical
    for col in numeric_cols:
        if col != target_col:
            median_val = df[col].median()
            df['_binary_group'] = (df[col] > median_val).map({True: 'high', False: 'low'})
            binary_col = '_binary_group'
            break

hook(binary_col, "binary grouping column", name='binary_col')
print(f"Grouping by: {binary_col}")

# Step 3: Get the two groups
groups = df[binary_col].dropna().unique()
group1_name, group2_name = groups[0], groups[1]
hook([str(group1_name), str(group2_name)], "group names", name='group_names', depends_on=['binary_col'])

# Step 4: Extract data for each group
group1_data = df[df[binary_col] == group1_name][target_col].dropna()
group2_data = df[df[binary_col] == group2_name][target_col].dropna()
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
    "mean1": round(float(mean1), 4),
    "mean2": round(float(mean2), 4),
    "t_statistic": round(float(t_stat), 4),
    "p_value": round(float(p_value), 6),
    "significant": is_significant
})
'''.strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 2,
    n_steps=8,
    difficulty="HARD",
)

BOOTSTRAP_CI_DISCOVERED = CompositionTemplate(
    name="bootstrap_ci_discovered",
    description="Bootstrap 95% CI for the mean of the column with highest skewness",
    output_schema='A JSON object with exactly 7 keys: "column" (name of most skewed column), "skewness" (rounded to 4 decimals), "mean" (rounded to 4 decimals), "ci_lower" (lower bound of 95% CI, rounded to 4 decimals), "ci_upper" (upper bound, rounded to 4 decimals), "std_error" (bootstrap standard error, rounded to 4 decimals), and "n_bootstrap" (integer, the number of bootstrap samples). Example: {"column": "income", "skewness": 2.3456, "mean": 50000.1234, "ci_lower": 48500.5678, "ci_upper": 51500.9012, "std_error": 750.3456, "n_bootstrap": 1000}',
    code_template='''
# Step 1: Identify numeric columns
numeric_cols = df.select_dtypes('number').columns.tolist()
hook(len(numeric_cols), "number of numeric columns", name='n_numeric')

# Step 2: Compute skewness for each column
skewness = {}
for col in numeric_cols:
    skew_val = df[col].skew()
    if not np.isnan(skew_val):
        skewness[col] = abs(skew_val)

hook(skewness, "absolute skewness per column", name='skewness')
print(f"Skewness values:\\n{sorted(skewness.items(), key=lambda x: -x[1])[:5]}")

# Step 3: Find column with highest absolute skewness
target_col = max(skewness, key=skewness.get)
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
np.random.seed(42)
n_bootstrap = 1000
bootstrap_means = []
for _ in range(n_bootstrap):
    sample = np.random.choice(data, size=n, replace=True)
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
    "skewness": round(float(skewness[target_col]), 4),
    "mean": round(float(original_mean), 4),
    "ci_lower": round(float(ci_lower), 4),
    "ci_upper": round(float(ci_upper), 4),
    "std_error": round(float(se), 4),
    "n_bootstrap": n_bootstrap
})
'''.strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 2,
    n_steps=9,
    difficulty="VERY_HARD",
)

ANOVA_DISCOVERED_GROUPS = CompositionTemplate(
    name="anova_discovered_groups",
    description="Find categorical column with 3+ groups, perform ANOVA on highest-variance numeric",
    output_schema='A JSON object with exactly 11 keys: "target_column", "grouping_column", "n_groups" (integer), "f_statistic" (rounded to 4 decimals), "p_value" (rounded to 6 decimals), "significant" (boolean), "best_group" (category with highest mean), "best_mean" (rounded to 4 decimals), "worst_group" (category with lowest mean), "worst_mean" (rounded to 4 decimals), and "eta_squared" (effect size, rounded to 4 decimals). Example: {"target_column": "sales", "grouping_column": "region", "n_groups": 4, "f_statistic": 15.2345, "p_value": 0.000012, "significant": true, "best_group": "West", "best_mean": 1234.5678, "worst_group": "East", "worst_mean": 890.1234, "eta_squared": 0.1523}',
    code_template='''
# Step 1: Find numeric column with highest variance
numeric_cols = df.select_dtypes('number').columns.tolist()
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
    submit({"error": "No suitable categorical column found (need 3-10 groups)"})
else:
    hook(group_col, "grouping column", name='group_col')
    print(f"Grouping by: {group_col}")

    # Step 3: Get group names and sizes
    group_counts = df[group_col].value_counts()
    n_groups = len(group_counts)
    hook({"n_groups": n_groups, "counts": group_counts.to_dict()}, "group info", name='group_info', depends_on=['group_col'])
    print(f"Groups: {n_groups}")

    # Step 4: Compute group means
    group_means = df.groupby(group_col)[target_col].mean()
    hook(group_means.to_dict(), "group means", name='group_means', depends_on=['group_info'])
    print(f"Group means:\\n{group_means.sort_values(ascending=False)}")

    # Step 5: Prepare data for ANOVA
    groups_data = [df[df[group_col] == g][target_col].dropna().values for g in group_counts.index]
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
    ss_between = sum(len(g) * (g.mean() - df[target_col].mean())**2 for g in groups_data)
    ss_total = ((df[target_col] - df[target_col].mean())**2).sum()
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    hook(eta_squared, "eta-squared effect size", name='effect_size', depends_on=['extremes'])

    submit({
        "target_column": target_col,
        "grouping_column": group_col,
        "n_groups": n_groups,
        "f_statistic": round(float(f_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": p_value < 0.05,
        "best_group": str(best_group),
        "best_mean": round(float(group_means[best_group]), 4),
        "worst_group": str(worst_group),
        "worst_mean": round(float(group_means[worst_group]), 4),
        "eta_squared": round(float(eta_squared), 4)
    })
'''.strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 1 and _has_categorical_cols(p),
    n_steps=9,
    difficulty="VERY_HARD",
)

MULTIPLE_REGRESSION_TOP_PREDICTORS = CompositionTemplate(
    name="multiple_regression_top_predictors",
    description="Find top 3 predictors via correlation, build multiple regression, report adjusted R-squared",
    output_schema='A JSON object with exactly 6 keys: "target" (column name), "predictors" (list of 3 column names), "r_squared" (rounded to 4 decimals), "adj_r_squared" (rounded to 4 decimals), "n_significant" (integer count of predictors with p < 0.05), "coefficients" (dict mapping predictor names to coefficients rounded to 4 decimals), and "p_values" (dict mapping predictor names to p-values rounded to 6 decimals). Example: {"target": "price", "predictors": ["sqft", "bedrooms", "age"], "r_squared": 0.7234, "adj_r_squared": 0.7156, "n_significant": 2, "coefficients": {"sqft": 123.45, "bedrooms": 5000.12, "age": -200.34}, "p_values": {"sqft": 0.000001, "bedrooms": 0.023456, "age": 0.156789}}',
    code_template='''
# Step 1: Identify numeric columns
numeric_cols = df.select_dtypes('number').columns.tolist()
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

    # Step 8: Extract coefficients and p-values
    coefs = {col: {"coef": float(model.params[col]), "p_value": float(model.pvalues[col])} for col in top_3}
    hook(coefs, "coefficients and p-values", name='coefficients', depends_on=['model_fit'])

    # Step 9: Count significant predictors
    n_significant = sum(1 for col in top_3 if model.pvalues[col] < 0.05)
    hook(n_significant, "significant predictors at alpha=0.05", name='n_significant', depends_on=['coefficients'])

    submit({
        "target": target_col,
        "predictors": top_3,
        "r_squared": round(float(model.rsquared), 4),
        "adj_r_squared": round(float(model.rsquared_adj), 4),
        "n_significant": n_significant,
        "coefficients": {col: round(coefs[col]["coef"], 4) for col in top_3},
        "p_values": {col: round(coefs[col]["p_value"], 6) for col in top_3}
    })
'''.strip(),
    output_type="dict",
    applicable_when=lambda p: _count_numeric_cols(p) >= 4,
    n_steps=10,
    difficulty="VERY_HARD",
)


# =============================================================================
# ALL TEMPLATES REGISTRY
# =============================================================================

ALL_TEMPLATES = [
    # Superlative (2) - 3 steps
    MAX_VARIANCE_MEAN,
    MIN_MEAN_COLUMN_STD,
    # Cross-column (2) - 4 steps
    STRONGEST_CORRELATION,
    WEAKEST_CORRELATION,
    # Conditional (1) - 5 steps
    CONDITIONAL_NORMALITY,
    # Threshold (2) - 3-4 steps
    COUNT_HIGH_MISSING_COLUMNS,
    COUNT_OUTLIER_COLUMNS,
    # Multi-stage (2) - 5 steps
    CATEGORY_WITH_HIGHEST_TARGET_MEAN,
    CORRELATION_AFTER_OUTLIER_REMOVAL,
    # Complex statistical (5) - 8-10 steps
    REGRESSION_MOST_PREDICTIVE,
    TTEST_DISCOVERED_GROUPS,
    BOOTSTRAP_CI_DISCOVERED,
    ANOVA_DISCOVERED_GROUPS,
    MULTIPLE_REGRESSION_TOP_PREDICTORS,
]


def get_applicable_templates(profile: dict) -> list[CompositionTemplate]:
    """Return templates that are applicable to the given dataset profile.

    Returns templates sorted by n_steps descending (hardest first).
    """
    applicable = [t for t in ALL_TEMPLATES if t.is_applicable(profile)]
    return sorted(applicable, key=lambda t: t.n_steps, reverse=True)

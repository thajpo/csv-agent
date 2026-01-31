"""Operator library for compositional program generation.

Rules:
- Operators are typed and have preconditions.
- No arbitrary "first column" selection in operators.
- All column binding happens via enumeration, not inside operators.
"""

from src.datagen.synthetic.programs.spec import Op, State

ID_PATTERNS = [
    r"^unnamed:\s*\d+$",
    r"^index$",
    r"^id$",
    r"^_id$",
    r"^row_?id$",
    r"^person_?id$",
]


def _is_id_like(col_name: str) -> bool:
    import re

    lowered = col_name.lower().strip()
    return any(re.match(p, lowered, re.IGNORECASE) for p in ID_PATTERNS)


def get_eligible_numeric_cols(profile: dict) -> list[str]:
    """Eligible numeric columns: numeric, non-ID, low missingness."""
    numeric = [
        col
        for col, info in profile.get("columns", {}).items()
        if info.get("type") == "numeric"
        and not _is_id_like(col)
        and info.get("missing_pct", 100) < 50
    ]
    return sorted(numeric)


def get_eligible_categorical_cols(profile: dict) -> list[str]:
    """Eligible categorical columns: categorical, non-ID, low missingness."""
    cat = [
        col
        for col, info in profile.get("columns", {}).items()
        if info.get("type") == "categorical"
        and not _is_id_like(col)
        and info.get("missing_pct", 100) < 50
        and info.get("unique_count", 0) >= 2
    ]
    return sorted(cat)


def get_eligible_binary_categorical_cols(profile: dict) -> list[str]:
    """Binary categorical columns with min support.

    Enumeration uses these columns and produces one program per column.
    """
    cols = []
    for col, info in profile.get("columns", {}).items():
        if info.get("type") == "categorical" and info.get("unique_count", 0) == 2:
            cols.append(col)
    return sorted(cols)


# Selectors
def select_numeric_cols_emit(state: State) -> str:
    return f"numeric_cols = {state.numeric_cols}"


def select_categorical_cols_emit(state: State) -> str:
    return f"categorical_cols = {state.categorical_cols}"


def bind_numeric_col_emit(state: State) -> str:
    col = state.bindings.get("selected_col")
    return f"selected_col = '{col}'"


def bind_binary_cat_col_emit(state: State) -> str:
    col = state.bindings.get("cat_col")
    return f"cat_col = '{col}'"


def bind_num_col_1_emit(state: State) -> str:
    """Emit code to bind the first numeric column.

    This operator binds a specific column to num_col_1 for use by
    multi-input operators like correlation.

    Why separate binding ops:
    - Enables explicit enumeration of column combinations
    - No arbitrary selection inside the analysis operator
    - Clear data flow: bind col1 → bind col2 → analyze
    """
    col = state.bindings.get("num_col_1")
    return f"num_col_1 = '{col}'"


def bind_num_col_2_emit(state: State) -> str:
    """Emit code to bind the second numeric column.

    Must be called after bind_num_col_1 and must bind a different column.
    The precondition ensures col1 != col2.
    """
    col = state.bindings.get("num_col_2")
    return f"num_col_2 = '{col}'"


# Property-based selectors (no arbitrary column choice)
def pick_numeric_by_variance_emit(state: State) -> str:
    strategy = state.bindings.get("strategy", "max")
    if strategy == "max":
        return "selected_col = df[numeric_cols].var().idxmax()"
    return "selected_col = df[numeric_cols].var().idxmin()"


def pick_numeric_by_skew_emit(state: State) -> str:
    strategy = state.bindings.get("strategy", "max")
    if strategy == "max":
        return "selected_col = df[numeric_cols].skew().idxmax()"
    return "selected_col = df[numeric_cols].skew().idxmin()"


def pick_categorical_by_cardinality_emit(state: State) -> str:
    strategy = state.bindings.get("strategy", "min")
    if strategy == "max":
        return "cat_col = df[categorical_cols].nunique().idxmax()"
    return "cat_col = df[categorical_cols].nunique().idxmin()"


# Transforms
def zscore_emit(state: State) -> str:
    return (
        "std_val = df[selected_col].std(ddof=0)\n"
        "std_val = std_val if std_val != 0 else 1.0\n"
        "num_series = (df[selected_col] - df[selected_col].mean()) / std_val"
    )


def log1p_emit(state: State) -> str:
    return "num_series = np.log1p(df[selected_col].clip(lower=0))"


def abs_emit(state: State) -> str:
    return "num_series = df[selected_col].abs()"


# Evidence
def shapiro_p_emit(state: State) -> str:
    return (
        "shapiro_stat, normality_p = scipy.stats.shapiro(df[selected_col].dropna())\n"
        'hook(normality_p, name="normality_p", code_line="shapiro_p")'
    )


def levene_p_emit(state: State) -> str:
    return "levene_stat, equal_var_p = scipy.stats.levene(*groups)"


# Decision
def choose_test_emit(state: State) -> str:
    return (
        'chosen_test = "ttest" if normality_p > 0.05 and equal_var_p > 0.05 else "mwu"\n'
        'hook(chosen_test, name="chosen_test", code_line="choose_test")'
    )


# Analysis
def groupby_values_emit(state: State) -> str:
    return "groups = [s for _, s in df.groupby(cat_col)[selected_col]]"


def groupby_mean_emit(state: State) -> str:
    return "group_means = df.groupby(cat_col)[selected_col].mean()"


def groupby_var_emit(state: State) -> str:
    return "group_vars = df.groupby(cat_col)[selected_col].var()"


def groupby_median_emit(state: State) -> str:
    return "group_medians = df.groupby(cat_col)[selected_col].median()"


def groupby_std_emit(state: State) -> str:
    return "group_stds = df.groupby(cat_col)[selected_col].std()"


def groupby_count_emit(state: State) -> str:
    return "group_counts = df.groupby(cat_col).size()"


def argmax_group_emit(state: State) -> str:
    return (
        "max_group = group_means.idxmax()\n"
        'submit({"group": str(max_group), "mean": round(group_means[max_group], 3)})'
    )


def argmin_group_emit(state: State) -> str:
    return (
        "min_group = group_means.idxmin()\n"
        'submit({"group": str(min_group), "mean": round(group_means[min_group], 3)})'
    )


def argmax_group_median_emit(state: State) -> str:
    return (
        "max_group = group_medians.idxmax()\n"
        'submit({"group": str(max_group), "median": round(group_medians[max_group], 3)})'
    )


def argmin_group_median_emit(state: State) -> str:
    return (
        "min_group = group_medians.idxmin()\n"
        'submit({"group": str(min_group), "median": round(group_medians[min_group], 3)})'
    )


def argmax_group_std_emit(state: State) -> str:
    return (
        "max_group = group_stds.idxmax()\n"
        'submit({"group": str(max_group), "std": round(group_stds[max_group], 3)})'
    )


def argmin_group_std_emit(state: State) -> str:
    return (
        "min_group = group_stds.idxmin()\n"
        'submit({"group": str(min_group), "std": round(group_stds[min_group], 3)})'
    )


def argmax_group_var_emit(state: State) -> str:
    return (
        "max_group = group_vars.idxmax()\n"
        'submit({"group": str(max_group), "variance": round(group_vars[max_group], 3)})'
    )


def argmin_group_var_emit(state: State) -> str:
    return (
        "min_group = group_vars.idxmin()\n"
        'submit({"group": str(min_group), "variance": round(group_vars[min_group], 3)})'
    )


def argmax_group_count_emit(state: State) -> str:
    return (
        "max_group = group_counts.idxmax()\n"
        'submit({"group": str(max_group), "count": int(group_counts[max_group])})'
    )


def argmin_group_count_emit(state: State) -> str:
    return (
        "min_group = group_counts.idxmin()\n"
        'submit({"group": str(min_group), "count": int(group_counts[min_group])})'
    )


def mean_emit(state: State) -> str:
    return (
        "mean_val = df[selected_col].mean()\n"
        'submit({"column": selected_col, "mean": round(mean_val, 3)})'
    )


def median_emit(state: State) -> str:
    return (
        "median_val = df[selected_col].median()\n"
        'submit({"column": selected_col, "median": round(median_val, 3)})'
    )


def std_emit(state: State) -> str:
    return (
        "std_val = df[selected_col].std()\n"
        'submit({"column": selected_col, "std": round(std_val, 3)})'
    )


def variance_emit(state: State) -> str:
    return (
        "var_val = df[selected_col].var()\n"
        'submit({"column": selected_col, "variance": round(var_val, 3)})'
    )


def mean_series_emit(state: State) -> str:
    return (
        "mean_val = num_series.mean()\n"
        'submit({"metric": "mean", "value": round(mean_val, 3)})'
    )


def median_series_emit(state: State) -> str:
    return (
        "median_val = num_series.median()\n"
        'submit({"metric": "median", "value": round(median_val, 3)})'
    )


def std_series_emit(state: State) -> str:
    return (
        "std_val = num_series.std()\n"
        'submit({"metric": "std", "value": round(std_val, 3)})'
    )


def max_series_emit(state: State) -> str:
    return (
        "max_val = num_series.max()\n"
        'submit({"metric": "max", "value": round(max_val, 3)})'
    )


def min_series_emit(state: State) -> str:
    return (
        "min_val = num_series.min()\n"
        'submit({"metric": "min", "value": round(min_val, 3)})'
    )


def ttest_ind_emit(state: State) -> str:
    return (
        'if chosen_test == "ttest":\n'
        "    stat, p_value = scipy.stats.ttest_ind(*groups)\n"
        "else:\n"
        "    stat, p_value = scipy.stats.mannwhitneyu(*groups)\n"
        'submit({"test": chosen_test, "stat": round(stat, 3), "p_value": round(p_value, 3)})'
    )


def correlation_emit(state: State) -> str:
    """Emit code to compute correlation between two bound columns.

    Uses num_col_1 and num_col_2 which must be bound by previous operators.
    This is the explicit multi-input approach: bindings are filled during
    enumeration, not selected arbitrarily inside the operator.

    Why this design:
    - All valid (col1, col2) pairs are enumerated systematically
    - No hidden selection logic
    - Reproducible: same bindings always produce same result
    - Compositional: can chain with other ops that use these columns
    """
    return (
        "# Compute correlation between explicitly bound columns\n"
        "corr = df[num_col_1].corr(df[num_col_2])\n"
        'submit({"col1": num_col_1, "col2": num_col_2, "correlation": round(float(corr), 3)})'
    )


def ratio_emit(state: State) -> str:
    """Emit code to compute ratio of two bound columns.

    Computes col1 / col2 for the two explicitly bound columns.
    This demonstrates another multi-input operator pattern.

    Why add this operator:
    - Shows the pattern generalizes beyond correlation
    - Creates questions about relative magnitudes
    - Can reveal interesting relationships in the data
    """
    return (
        "# Compute ratio of explicitly bound columns\n"
        "ratio = df[num_col_1] / df[num_col_2].replace(0, np.nan)\n"
        "mean_ratio = ratio.mean()\n"
        'submit({"col1": num_col_1, "col2": num_col_2, "mean_ratio": round(float(mean_ratio), 3)})'
    )


def filter_greater_than_emit(state: State) -> str:
    """Emit code to filter rows where column > threshold.

    This is a transform operator that reduces the dataset.
    The filtered dataframe can be used by subsequent operators.

    Why filtering matters:
    - Enables conditional analysis (e.g., "for high values of X...")
    - Creates more specific, targeted questions
    - Can be chained: filter → group → aggregate

    Design decision: We emit code that creates a filtered view,
    but we don't overwrite df to allow flexible chaining.
    """
    threshold = state.bindings.get("threshold", 0)
    return (
        f"# Filter rows where {state.bindings.get('selected_col', 'selected_col')} > {threshold}\n"
        f"filtered = df[df['{state.bindings.get('selected_col', 'selected_col')}'] > {threshold}]\n"
        f"n_filtered = len(filtered)\n"
        f"hook(n_filtered, 'rows after filtering', name='n_filtered')\n"
    )


def min_emit(state: State) -> str:
    """Emit code to compute minimum of a bound column."""
    return (
        "min_val = df[num_col_1].min()\n"
        'submit({"column": num_col_1, "min": round(float(min_val), 3)})'
    )


def max_emit(state: State) -> str:
    """Emit code to compute maximum of a bound column."""
    return (
        "max_val = df[num_col_1].max()\n"
        'submit({"column": num_col_1, "max": round(float(max_val), 3)})'
    )


def sum_emit(state: State) -> str:
    return (
        "total = df[selected_col].sum()\n"
        'submit({"column": selected_col, "sum": round(float(total), 3)})'
    )


# New chainable operators for 10-15 step complexity
def filter_by_threshold_emit(state: State) -> str:
    """Emit code to filter table by threshold on selected column.

    This operator consumes and produces Table, enabling chaining.
    Example chain: filter > threshold → sort → top_n → aggregate
    """
    col = state.bindings.get("selected_col", "selected_col")
    threshold = state.bindings.get("threshold", 0)
    return (
        f"# Filter rows where {col} > {threshold}\n"
        f"df = df[df['{col}'] > {threshold}].reset_index(drop=True)\n"
        f"n_rows = len(df)\n"
        f"hook(n_rows, 'rows after filter', name='n_rows')"
    )


def sort_by_column_emit(state: State) -> str:
    """Emit code to sort table by selected column.

    Enables ranking chains: sort → top_n → aggregate
    """
    col = state.bindings.get("selected_col", "selected_col")
    ascending = state.bindings.get("ascending", True)
    return (
        f"# Sort by {col} (ascending={ascending})\n"
        f"df = df.sort_values('{col}', ascending={ascending}).reset_index(drop=True)"
    )


def top_n_emit(state: State) -> str:
    """Emit code to take top N rows.

    After sorting, take top N for focused analysis.
    """
    n = state.bindings.get("n", 10)
    return (
        f"# Take top {n} rows\n"
        f"df = df.head({n}).reset_index(drop=True)\n"
        f"hook({n}, 'selected top n', name='top_n')"
    )


def cumulative_sum_emit(state: State) -> str:
    """Emit code to compute cumulative sum.

    Chain: bind column → cumsum → analyze trend
    """
    col = state.bindings.get("selected_col", "selected_col")
    return (
        f"# Compute cumulative sum of {col}\n"
        f"cumsum_col = '{col}_cumsum'\n"
        f"df[cumsum_col] = df['{col}'].cumsum()\n"
        f"final_cumsum = df[cumsum_col].iloc[-1]\n"
        f"hook(final_cumsum, 'cumulative sum', name='cumsum')"
    )


def rolling_mean_emit(state: State) -> str:
    """Emit code to compute rolling mean.

    Windowed analysis for trends.
    """
    col = state.bindings.get("selected_col", "selected_col")
    window = state.bindings.get("window", 3)
    return (
        f"# Compute {window}-period rolling mean of {col}\n"
        f"rolling_col = '{col}_rolling_{window}'\n"
        f"df[rolling_col] = df['{col}'].rolling(window={window}, min_periods=1).mean()\n"
        f"mean_rolling = df[rolling_col].mean()\n"
        f"hook(mean_rolling, 'rolling mean', name='rolling_mean')"
    )


def diff_emit(state: State) -> str:
    """Emit code to compute difference from previous row.

    For change detection chains.
    """
    col = state.bindings.get("selected_col", "selected_col")
    return (
        f"# Compute difference of {col}\n"
        f"diff_col = '{col}_diff'\n"
        f"df[diff_col] = df['{col}'].diff().fillna(0)\n"
        f"max_diff = df[diff_col].abs().max()\n"
        f"hook(max_diff, 'max difference', name='max_diff')"
    )


def percentile_rank_emit(state: State) -> str:
    """Emit code to compute percentile rank within column.

    Ranking analysis.
    """
    col = state.bindings.get("selected_col", "selected_col")
    return (
        f"# Compute percentile rank of {col}\n"
        f"rank_col = '{col}_percentile'\n"
        f"df[rank_col] = df['{col}'].rank(pct=True)\n"
        f"mean_percentile = df[rank_col].mean()\n"
        f"hook(mean_percentile, 'mean percentile', name='mean_percentile')"
    )


def bin_into_quartiles_emit(state: State) -> str:
    """Emit code to bin column into quartiles.

    Categorization for grouped analysis.
    """
    col = state.bindings.get("selected_col", "selected_col")
    return (
        f"# Bin {col} into quartiles\n"
        f"quartile_col = '{col}_quartile'\n"
        f"df[quartile_col] = pd.qcut(df['{col}'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])\n"
        f"quartile_counts = df[quartile_col].value_counts().to_dict()\n"
        f"hook(str(quartile_counts), 'quartile distribution', name='quartiles')"
    )


OPERATORS = {
    "select_numeric_cols": Op(
        name="select_numeric_cols",
        inputs=["Table"],
        outputs=["NumCols"],
        attributes=["selector"],
        emit=select_numeric_cols_emit,
        update=lambda s: setattr(
            s, "numeric_cols", get_eligible_numeric_cols(s.bindings.get("profile", {}))
        )
        or None,
        precondition=lambda profile, _s: len(get_eligible_numeric_cols(profile)) > 0,
        reads=set(),
        writes={"numeric_cols"},
        requires_bindings={},
        produces=["numeric_cols"],
        consumes=[],
    ),
    "select_categorical_cols": Op(
        name="select_categorical_cols",
        inputs=["Table"],
        outputs=["CatCols"],
        attributes=["selector"],
        emit=select_categorical_cols_emit,
        update=lambda s: setattr(
            s,
            "categorical_cols",
            get_eligible_categorical_cols(s.bindings.get("profile", {})),
        )
        or None,
        precondition=lambda profile, _s: len(get_eligible_categorical_cols(profile))
        > 0,
        reads=set(),
        writes={"categorical_cols"},
        requires_bindings={},
        produces=["categorical_cols"],
        consumes=[],
    ),
    "bind_numeric_col": Op(
        name="bind_numeric_col",
        inputs=["NumCols"],
        outputs=["NumCol"],
        attributes=["selector"],
        emit=bind_numeric_col_emit,
        update=lambda s: None,
        precondition=lambda _profile, s: len(s.numeric_cols) > 0,
        reads={"numeric_cols"},
        writes={"selected_col"},
        requires_bindings={"selected_col": True},
        produces=["selected_col"],
        consumes=["numeric_cols"],
    ),
    "bind_binary_cat_col": Op(
        name="bind_binary_cat_col",
        inputs=["CatCols"],
        outputs=["CatCol"],
        attributes=["selector"],
        emit=bind_binary_cat_col_emit,
        update=lambda s: None,
        precondition=lambda _profile, s: len(s.categorical_cols) > 0,
        reads={"categorical_cols"},
        writes={"cat_col"},
        requires_bindings={"cat_col": True},
        produces=["cat_col"],
        consumes=["categorical_cols"],
    ),
    # Multi-input binding operators
    # These enable explicit enumeration of column combinations for ops like correlation
    "bind_num_col_1": Op(
        name="bind_num_col_1",
        inputs=["NumCols"],
        outputs=["NumCol"],  # Adds 1 NumCol to the type count
        attributes=["selector"],
        emit=bind_num_col_1_emit,
        update=lambda s: None,
        precondition=lambda _profile, s: len(s.numeric_cols) > 0,
        reads={"numeric_cols"},
        writes={"num_col_1"},
        requires_bindings={"num_col_1": True},
        produces=["num_col_1"],
        consumes=["numeric_cols"],
    ),
    "bind_num_col_2": Op(
        name="bind_num_col_2",
        inputs=["NumCols"],  # Still needs NumCols as input (to know available columns)
        outputs=["NumCol"],  # Adds another NumCol (now we have 2)
        attributes=["selector"],
        emit=bind_num_col_2_emit,
        update=lambda s: None,
        # Precondition: must have at least 2 columns available
        # Note: We don't check col1 != col2 here because bindings are empty during
        # grammar search. The distinctness constraint is enforced during enumeration
        # when we generate all valid (col1, col2) pairs where col1 != col2.
        # This separation allows the grammar to find the structural pattern,
        # while enumeration ensures semantic validity.
        precondition=lambda _profile, s: len(s.numeric_cols) > 1,
        reads={"numeric_cols"},  # Only reads available columns, not specific bindings
        writes={"num_col_2"},
        requires_bindings={"num_col_2": True},
        produces=["num_col_2"],
        consumes=["numeric_cols"],
    ),
    "pick_numeric_by_variance": Op(
        name="pick_numeric_by_variance",
        inputs=["NumCols"],
        outputs=["NumCol"],
        attributes=["selector"],
        emit=pick_numeric_by_variance_emit,
        update=lambda s: None,
        precondition=lambda _profile, s: len(s.numeric_cols) > 0,
        reads={"numeric_cols"},
        writes={"selected_col"},
        requires_bindings={},
        produces=["selected_col"],
        consumes=["numeric_cols"],
    ),
    "pick_numeric_by_skew": Op(
        name="pick_numeric_by_skew",
        inputs=["NumCols"],
        outputs=["NumCol"],
        attributes=["selector"],
        emit=pick_numeric_by_skew_emit,
        update=lambda s: None,
        precondition=lambda _profile, s: len(s.numeric_cols) > 0,
        reads={"numeric_cols"},
        writes={"selected_col"},
        requires_bindings={},
        produces=["selected_col"],
        consumes=["numeric_cols"],
    ),
    "pick_categorical_by_cardinality": Op(
        name="pick_categorical_by_cardinality",
        inputs=["CatCols"],
        outputs=["CatCol"],
        attributes=["selector"],
        emit=pick_categorical_by_cardinality_emit,
        update=lambda s: None,
        precondition=lambda _profile, s: len(s.categorical_cols) > 0,
        reads={"categorical_cols"},
        writes={"cat_col"},
        requires_bindings={},
        produces=["cat_col"],
        consumes=["categorical_cols"],
    ),
    "zscore": Op(
        name="zscore",
        inputs=["NumCol"],
        outputs=["NumSeries"],
        attributes=["transform"],
        emit=zscore_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"selected_col"},
        writes={"num_series"},
        requires_bindings={},
        produces=["num_series"],
        consumes=["selected_col"],
    ),
    "log1p": Op(
        name="log1p",
        inputs=["NumCol"],
        outputs=["NumSeries"],
        attributes=["transform"],
        emit=log1p_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"selected_col"},
        writes={"num_series"},
        requires_bindings={},
        produces=["num_series"],
        consumes=["selected_col"],
    ),
    "abs": Op(
        name="abs",
        inputs=["NumCol"],
        outputs=["NumSeries"],
        attributes=["transform"],
        emit=abs_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"selected_col"},
        writes={"num_series"},
        requires_bindings={},
        produces=["num_series"],
        consumes=["selected_col"],
    ),
    "groupby_mean": Op(
        name="groupby_mean",
        inputs=["Table", "CatCol", "NumCol"],
        outputs=["Dict"],
        attributes=["analysis"],
        emit=groupby_mean_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"cat_col", "selected_col"},
        writes={"group_means"},
        requires_bindings={},
        produces=["group_means"],
        consumes=["cat_col", "selected_col"],
    ),
    "groupby_var": Op(
        name="groupby_var",
        inputs=["Table", "CatCol", "NumCol"],
        outputs=["Dict"],
        attributes=["analysis"],
        emit=groupby_var_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"cat_col", "selected_col"},
        writes={"group_vars"},
        requires_bindings={},
        produces=["group_vars"],
        consumes=["cat_col", "selected_col"],
    ),
    "groupby_median": Op(
        name="groupby_median",
        inputs=["Table", "CatCol", "NumCol"],
        outputs=["Dict"],
        attributes=["analysis"],
        emit=groupby_median_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"cat_col", "selected_col"},
        writes={"group_medians"},
        requires_bindings={},
        produces=["group_medians"],
        consumes=["cat_col", "selected_col"],
    ),
    "groupby_std": Op(
        name="groupby_std",
        inputs=["Table", "CatCol", "NumCol"],
        outputs=["Dict"],
        attributes=["analysis"],
        emit=groupby_std_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"cat_col", "selected_col"},
        writes={"group_stds"},
        requires_bindings={},
        produces=["group_stds"],
        consumes=["cat_col", "selected_col"],
    ),
    "groupby_count": Op(
        name="groupby_count",
        inputs=["Table", "CatCol"],
        outputs=["Dict"],
        attributes=["analysis"],
        emit=groupby_count_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"cat_col"},
        writes={"group_counts"},
        requires_bindings={},
        produces=["group_counts"],
        consumes=["cat_col"],
    ),
    "argmax_group": Op(
        name="argmax_group",
        inputs=["Dict"],
        outputs=["Dict"],
        attributes=["analysis"],
        emit=argmax_group_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"group_means"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["group_means"],
    ),
    "argmin_group": Op(
        name="argmin_group",
        inputs=["Dict"],
        outputs=["Dict"],
        attributes=["analysis"],
        emit=argmin_group_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"group_means"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["group_means"],
    ),
    "argmax_group_median": Op(
        name="argmax_group_median",
        inputs=["Dict"],
        outputs=["Dict"],
        attributes=["analysis"],
        emit=argmax_group_median_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"group_medians"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["group_medians"],
    ),
    "argmin_group_median": Op(
        name="argmin_group_median",
        inputs=["Dict"],
        outputs=["Dict"],
        attributes=["analysis"],
        emit=argmin_group_median_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"group_medians"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["group_medians"],
    ),
    "argmax_group_std": Op(
        name="argmax_group_std",
        inputs=["Dict"],
        outputs=["Dict"],
        attributes=["analysis"],
        emit=argmax_group_std_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"group_stds"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["group_stds"],
    ),
    "argmin_group_std": Op(
        name="argmin_group_std",
        inputs=["Dict"],
        outputs=["Dict"],
        attributes=["analysis"],
        emit=argmin_group_std_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"group_stds"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["group_stds"],
    ),
    "argmax_group_var": Op(
        name="argmax_group_var",
        inputs=["Dict"],
        outputs=["Dict"],
        attributes=["analysis"],
        emit=argmax_group_var_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"group_vars"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["group_vars"],
    ),
    "argmin_group_var": Op(
        name="argmin_group_var",
        inputs=["Dict"],
        outputs=["Dict"],
        attributes=["analysis"],
        emit=argmin_group_var_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"group_vars"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["group_vars"],
    ),
    "argmax_group_count": Op(
        name="argmax_group_count",
        inputs=["Dict"],
        outputs=["Dict"],
        attributes=["analysis"],
        emit=argmax_group_count_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"group_counts"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["group_counts"],
    ),
    "argmin_group_count": Op(
        name="argmin_group_count",
        inputs=["Dict"],
        outputs=["Dict"],
        attributes=["analysis"],
        emit=argmin_group_count_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"group_counts"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["group_counts"],
    ),
    "mean": Op(
        name="mean",
        inputs=["NumCol"],
        outputs=["Scalar"],
        attributes=["analysis"],
        emit=mean_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"selected_col"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["selected_col"],
    ),
    "median": Op(
        name="median",
        inputs=["NumCol"],
        outputs=["Scalar"],
        attributes=["analysis"],
        emit=median_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"selected_col"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["selected_col"],
    ),
    "std": Op(
        name="std",
        inputs=["NumCol"],
        outputs=["Scalar"],
        attributes=["analysis"],
        emit=std_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"selected_col"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["selected_col"],
    ),
    "variance": Op(
        name="variance",
        inputs=["NumCol"],
        outputs=["Scalar"],
        attributes=["analysis"],
        emit=variance_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"selected_col"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["selected_col"],
    ),
    "mean_series": Op(
        name="mean_series",
        inputs=["NumSeries"],
        outputs=["Scalar"],
        attributes=["analysis"],
        emit=mean_series_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"num_series"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["num_series"],
    ),
    "median_series": Op(
        name="median_series",
        inputs=["NumSeries"],
        outputs=["Scalar"],
        attributes=["analysis"],
        emit=median_series_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"num_series"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["num_series"],
    ),
    "std_series": Op(
        name="std_series",
        inputs=["NumSeries"],
        outputs=["Scalar"],
        attributes=["analysis"],
        emit=std_series_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"num_series"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["num_series"],
    ),
    "max_series": Op(
        name="max_series",
        inputs=["NumSeries"],
        outputs=["Scalar"],
        attributes=["analysis"],
        emit=max_series_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"num_series"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["num_series"],
    ),
    "min_series": Op(
        name="min_series",
        inputs=["NumSeries"],
        outputs=["Scalar"],
        attributes=["analysis"],
        emit=min_series_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"num_series"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["num_series"],
    ),
    "groupby_values": Op(
        name="groupby_values",
        inputs=["Table", "CatCol", "NumCol"],
        outputs=["Groups"],
        attributes=["analysis"],
        emit=groupby_values_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"cat_col", "selected_col"},
        writes={"groups"},
        requires_bindings={},
        produces=["groups"],
        consumes=["cat_col", "selected_col"],
    ),
    "shapiro_p": Op(
        name="shapiro_p",
        inputs=["NumCol"],
        outputs=["Scalar"],
        attributes=["evidence"],
        emit=shapiro_p_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"selected_col"},
        writes={"normality_p"},
        requires_bindings={},
        produces=["normality_p"],
        consumes=["selected_col"],
    ),
    "levene_p": Op(
        name="levene_p",
        inputs=["Groups"],
        outputs=["Scalar"],
        attributes=["evidence"],
        emit=levene_p_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"groups"},
        writes={"equal_var_p"},
        requires_bindings={},
        produces=["equal_var_p"],
        consumes=["groups"],
    ),
    "choose_test": Op(
        name="choose_test",
        inputs=["Scalar", "Scalar"],
        outputs=["Dict"],
        attributes=["decision"],
        emit=choose_test_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"normality_p", "equal_var_p"},
        writes={"chosen_test"},
        requires_bindings={},
        produces=["chosen_test"],
        consumes=["normality_p", "equal_var_p"],
    ),
    "ttest_ind": Op(
        name="ttest_ind",
        inputs=["Groups"],
        outputs=["Dict"],
        attributes=["test"],
        emit=ttest_ind_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"groups", "chosen_test"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["groups", "chosen_test"],
    ),
    # Correlation operator - uses two explicitly bound columns
    # Requires: bind_num_col_1 → bind_num_col_2 → correlation
    # Type signature: NumCol × 2 → Dict (uses count-based type checking)
    "correlation": Op(
        name="correlation",
        inputs=["NumCol", "NumCol"],  # Needs 2 NumCols (count-based)
        outputs=["Dict"],
        attributes=["analysis"],
        emit=correlation_emit,
        update=lambda s: None,
        # Precondition: Always true during grammar search
        # The distinctness of columns (col1 != col2) is enforced during enumeration
        # when we generate binding combinations. We skip this check here because
        # bindings are empty during grammar search (structure validation phase).
        # Separation of concerns:
        #   - Grammar: validates structural feasibility (types, operator order)
        #   - Enumeration: validates semantic constraints (distinct columns, etc.)
        precondition=lambda _profile, _s: True,
        reads={"num_col_1", "num_col_2"},  # Reads both bound columns
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},  # Bindings provided by previous ops
        produces=["answer"],
        consumes=["num_col_1", "num_col_2"],
    ),
    "filter_greater_than": Op(
        name="filter_greater_than",
        inputs=["Table", "NumCol"],
        outputs=["Table"],
        attributes=["transform"],
        emit=filter_greater_than_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"selected_col"},
        writes={"filtered_df"},
        requires_bindings={"threshold": False},
        produces=["filtered_df"],
        consumes=["selected_col"],
    ),
    "sum": Op(
        name="sum",
        inputs=["NumCol"],
        outputs=["Dict"],
        attributes=["analysis"],
        emit=sum_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"selected_col"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["selected_col"],
    ),
    # Additional multi-input operators demonstrating extensibility
    # These show how the Counter-based type system enables rich composition
    "ratio": Op(
        name="ratio",
        inputs=["NumCol", "NumCol"],  # Needs 2 NumCols (count-based)
        outputs=["Dict"],
        attributes=["analysis"],
        emit=ratio_emit,
        update=lambda s: None,
        # Same pattern as correlation: structural validation in grammar,
        # semantic validation (distinct columns) in enumeration
        precondition=lambda _profile, _s: True,
        reads={"num_col_1", "num_col_2"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["num_col_1", "num_col_2"],
    ),
    # Additional aggregation operators using explicit binding
    "min": Op(
        name="min",
        inputs=["NumCol"],
        outputs=["Dict"],
        attributes=["analysis"],
        emit=min_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"num_col_1"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["num_col_1"],
    ),
    "max": Op(
        name="max",
        inputs=["NumCol"],
        outputs=["Dict"],
        attributes=["analysis"],
        emit=max_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"num_col_1"},
        writes={"answer"},
        emits_answer=True,
        requires_bindings={},
        produces=["answer"],
        consumes=["num_col_1"],
    ),
    # Chainable operators for 10-15 step complexity
    # These operators consume and produce Table, enabling long chains
    "filter_by_threshold": Op(
        name="filter_by_threshold",
        inputs=["Table", "NumCol"],
        outputs=["Table"],  # Returns modified table for chaining
        attributes=["transform"],
        emit=filter_by_threshold_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"selected_col"},
        writes=set(),  # Modifies df in place
        requires_bindings={"threshold": False},
        produces=["df"],  # Modifies df in place
        consumes=["selected_col"],
    ),
    "sort_by_column": Op(
        name="sort_by_column",
        inputs=["Table", "NumCol"],
        outputs=["Table"],
        attributes=["transform"],
        emit=sort_by_column_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"selected_col"},
        writes=set(),
        requires_bindings={"ascending": False},
        produces=["df"],  # Modifies df in place
        consumes=["selected_col"],
    ),
    "top_n": Op(
        name="top_n",
        inputs=["Table"],
        outputs=["Table"],
        attributes=["transform"],
        emit=top_n_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads=set(),
        writes=set(),
        requires_bindings={"n": False},
        produces=["df"],  # Modifies df in place
        consumes=[],
    ),
    "cumulative_sum": Op(
        name="cumulative_sum",
        inputs=["Table", "NumCol"],
        outputs=["Table"],
        attributes=["transform"],
        emit=cumulative_sum_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"selected_col"},
        writes=set(),
        requires_bindings={},
        produces=["cumsum_col", "df"],  # Creates new column + modifies df
        consumes=["selected_col"],
    ),
    "rolling_mean": Op(
        name="rolling_mean",
        inputs=["Table", "NumCol"],
        outputs=["Table"],
        attributes=["transform"],
        emit=rolling_mean_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"selected_col"},
        writes=set(),
        requires_bindings={"window": False},
        produces=["rolling_col", "df"],  # Creates new column + modifies df
        consumes=["selected_col"],
    ),
    "diff": Op(
        name="diff",
        inputs=["Table", "NumCol"],
        outputs=["Table"],
        attributes=["transform"],
        emit=diff_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"selected_col"},
        writes=set(),
        requires_bindings={},
        produces=["diff_col", "df"],  # Creates new column + modifies df
        consumes=["selected_col"],
    ),
    "percentile_rank": Op(
        name="percentile_rank",
        inputs=["Table", "NumCol"],
        outputs=["Table"],
        attributes=["transform"],
        emit=percentile_rank_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"selected_col"},
        writes=set(),
        requires_bindings={},
        produces=["rank_col", "df"],  # Creates new column + modifies df
        consumes=["selected_col"],
    ),
    "bin_into_quartiles": Op(
        name="bin_into_quartiles",
        inputs=["Table", "NumCol"],
        outputs=["Table"],
        attributes=["transform"],
        emit=bin_into_quartiles_emit,
        update=lambda s: None,
        precondition=lambda _profile, _s: True,
        reads={"selected_col"},
        writes=set(),
        requires_bindings={},
        produces=["quartile_col", "df"],  # Creates new column + modifies df
        consumes=["selected_col"],
    ),
}


def get_operator(name: str) -> Op | None:
    return OPERATORS.get(name)

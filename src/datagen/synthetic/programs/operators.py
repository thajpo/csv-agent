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
    ),
}


def get_operator(name: str) -> Op | None:
    return OPERATORS.get(name)

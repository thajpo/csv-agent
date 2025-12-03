import json
import pandas as pd

# Available tools and their specs for prompt generation
# summarize: True = output may be large, should be summarized for context
#            False = output is atomic/small, keep as-is
TOOL_SPECS = {
    "inspect": {
        "name": "inspect",
        "type": "function",
        "description": "View dataframe structure (head, tail, shape, dtypes, columns, missing)",
        "parameters": {
            "type": "object",
            "properties": {
                "aspect": {
                    "type": "string",
                    "description": "The aspect of the dataframe to inspect",
                    "enum": ["head", "tail", "shape", "dtypes", "columns", "missing"]
                },
                "n": {
                    "type": "integer",
                    "description": "The number of rows to inspect",
                    "default": 5
                }
            },
            "required": ["aspect"],
            "additionalProperties": False
        },
        "summarize": True
    },
    "describe": {
        "name": "describe",
        "type": "function",
        "description": "Statistical summary of columns",
        "parameters": {
            "type": "object",
            "properties": {
                "include": {
                    "type": "string",
                    "description": "Type of columns to include in summary",
                    "enum": ["number", "object", "all"],
                    "default": "number"
                }
            },
            "required": [],
            "additionalProperties": False
        },
        "summarize": True
    },
    "value_counts": {
        "name": "value_counts",
        "type": "function",
        "description": "Frequency counts for a column",
        "parameters": {
            "type": "object",
            "properties": {
                "col": {
                    "type": "string",
                    "description": "Column name to count values for"
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top values to return",
                    "default": 20
                }
            },
            "required": ["col"],
            "additionalProperties": False
        },
        "summarize": True
    },
    "unique": {
        "name": "unique",
        "type": "function",
        "description": "List unique values in a column",
        "parameters": {
            "type": "object",
            "properties": {
                "col": {
                    "type": "string",
                    "description": "Column name to get unique values from"
                },
                "top_n": {
                    "type": "integer",
                    "description": "Maximum number of unique values to return",
                    "default": 50
                }
            },
            "required": ["col"],
            "additionalProperties": False
        },
        "summarize": True
    },
    "group_stat": {
        "name": "group_stat",
        "type": "function",
        "description": "Aggregate a column grouped by another. Add group_val for ONE group's scalar (atomic for hooks).",
        "parameters": {
            "type": "object",
            "properties": {
                "group_col": {
                    "type": "string",
                    "description": "Column to group by"
                },
                "target_col": {
                    "type": "string",
                    "description": "Column to aggregate"
                },
                "agg": {
                    "type": "string",
                    "description": "Aggregation function",
                    "enum": ["mean", "sum", "median", "std", "min", "max", "count", "nunique"],
                    "default": "mean"
                },
                "filter_expr": {
                    "type": "string",
                    "description": "Pandas query expression to filter rows",
                    "default": ""
                },
                "group_val": {
                    "type": "string",
                    "description": "Specific group value for atomic scalar output (optional)"
                }
            },
            "required": ["group_col", "target_col"],
            "additionalProperties": False
        },
        "summarize": False
    },
    "group_extremum": {
        "name": "group_extremum",
        "type": "function",
        "description": "Find which group has max/min aggregated value. Returns group name OR value (atomic).",
        "parameters": {
            "type": "object",
            "properties": {
                "group_col": {
                    "type": "string",
                    "description": "Column to group by"
                },
                "target_col": {
                    "type": "string",
                    "description": "Column to aggregate"
                },
                "agg": {
                    "type": "string",
                    "description": "Aggregation function",
                    "enum": ["mean", "sum", "median", "std", "min", "max", "count", "nunique"],
                    "default": "mean"
                },
                "extremum": {
                    "type": "string",
                    "description": "Find maximum or minimum",
                    "enum": ["max", "min"],
                    "default": "max"
                },
                "return_what": {
                    "type": "string",
                    "description": "Return group name or value",
                    "enum": ["group", "value"],
                    "default": "group"
                },
                "filter_expr": {
                    "type": "string",
                    "description": "Pandas query expression to filter rows",
                    "default": ""
                }
            },
            "required": ["group_col", "target_col"],
            "additionalProperties": False
        },
        "summarize": False
    },
    "correlation": {
        "name": "correlation",
        "type": "function",
        "description": "Correlation between two numeric columns",
        "parameters": {
            "type": "object",
            "properties": {
                "col_a": {
                    "type": "string",
                    "description": "First column name"
                },
                "col_b": {
                    "type": "string",
                    "description": "Second column name"
                },
                "method": {
                    "type": "string",
                    "description": "Correlation method",
                    "enum": ["pearson", "spearman", "kendall"],
                    "default": "pearson"
                },
                "filter_expr": {
                    "type": "string",
                    "description": "Pandas query expression to filter rows",
                    "default": ""
                }
            },
            "required": ["col_a", "col_b"],
            "additionalProperties": False
        },
        "summarize": False
    },
    "count_filter": {
        "name": "count_filter",
        "type": "function",
        "description": "Count rows matching a condition",
        "parameters": {
            "type": "object",
            "properties": {
                "filter_expr": {
                    "type": "string",
                    "description": "Pandas query expression to filter rows (empty string = all rows)",
                    "default": ""
                }
            },
            "required": [],
            "additionalProperties": False
        },
        "summarize": False
    },
    "sort_values": {
        "name": "sort_values",
        "type": "function",
        "description": "Sort by a column and show top rows",
        "parameters": {
            "type": "object",
            "properties": {
                "col": {
                    "type": "string",
                    "description": "Column to sort by"
                },
                "ascending": {
                    "type": "boolean",
                    "description": "Sort in ascending order",
                    "default": True
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top rows to return",
                    "default": 20
                }
            },
            "required": ["col"],
            "additionalProperties": False
        },
        "summarize": True
    },
    "quantile": {
        "name": "quantile",
        "type": "function",
        "description": "Calculate percentile(s) for a column. Single q = atomic scalar output.",
        "parameters": {
            "type": "object",
            "properties": {
                "col": {
                    "type": "string",
                    "description": "Column name"
                },
                "q": {
                    "oneOf": [
                        {"type": "number", "description": "Single quantile value (atomic output)"},
                        {"type": "array", "items": {"type": "number"}, "description": "List of quantiles"}
                    ],
                    "description": "Quantile value(s) to calculate"
                },
                "filter_expr": {
                    "type": "string",
                    "description": "Pandas query expression to filter rows",
                    "default": ""
                }
            },
            "required": ["col", "q"],
            "additionalProperties": False
        },
        "summarize": False
    },
    "crosstab": {
        "name": "crosstab",
        "type": "function",
        "description": "Cross-tabulation of two categorical columns",
        "parameters": {
            "type": "object",
            "properties": {
                "col_a": {
                    "type": "string",
                    "description": "First column name"
                },
                "col_b": {
                    "type": "string",
                    "description": "Second column name"
                },
                "normalize": {
                    "type": "string",
                    "description": "Normalization mode",
                    "enum": ["index", "columns", "all", ""],
                    "default": ""
                }
            },
            "required": ["col_a", "col_b"],
            "additionalProperties": False
        },
        "summarize": True
    },
    "derive_stat": {
        "name": "derive_stat",
        "type": "function",
        "description": "Compute derived metric (e.g. TL/IN), aggregate by group. Add group_val for atomic scalar.",
        "parameters": {
            "type": "object",
            "properties": {
                "formula": {
                    "type": "string",
                    "description": "Expression using column names (e.g., 'TL / IN')"
                },
                "group_col": {
                    "type": "string",
                    "description": "Column to group by"
                },
                "agg": {
                    "type": "string",
                    "description": "Aggregation function",
                    "enum": ["mean", "sum", "median", "std", "min", "max", "count"],
                    "default": "mean"
                },
                "filter_expr": {
                    "type": "string",
                    "description": "Pandas query expression to filter rows",
                    "default": ""
                },
                "group_val": {
                    "type": "string",
                    "description": "Specific group value for atomic scalar output (optional)"
                }
            },
            "required": ["formula", "group_col"],
            "additionalProperties": False
        },
        "summarize": False
    }
}


def should_summarize(tool_name: str) -> bool:
    """Check if a tool's output should be summarized."""
    return TOOL_SPECS.get(tool_name, {}).get("summarize", False)


def parse_tool_call(code: str) -> tuple[str, dict] | str:
    """
    Parse a JSON tool call from a code block.
    
    Returns:
        (tool_name, params) on success, or error string on failure.
    """
    try:
        data = json.loads(code.strip())
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"
    
    if not isinstance(data, dict):
        return "Tool call must be a JSON object"
    
    if "tool" not in data:
        return "Missing 'tool' field in JSON"
    
    tool = data.pop("tool")
    if tool not in TOOL_SPECS:
        available = ", ".join(TOOL_SPECS.keys())
        return f"Unknown tool: '{tool}'. Available: {available}"
    
    return (tool, data)


def run_tool(tool: str, df: pd.DataFrame, params: dict) -> str:
    """
    Run a tool on the dataframe and return the result.
    
    Uses .get() with defaults for optional parameters.
    """
    try:
        match tool:
            case "inspect":
                return inspect(df, params["aspect"], params.get("n", 5))
            case "describe":
                return describe(df, params.get("include", "number"))
            case "value_counts":
                return value_counts(df, params["col"], params.get("top_n", 20))
            case "unique":
                return unique(df, params["col"], params.get("top_n", 50))
            case "group_stat":
                return group_stat(
                    df,
                    params["group_col"],
                    params["target_col"],
                    params.get("agg", "mean"),
                    params.get("filter_expr", ""),
                    params.get("group_val"),
                )
            case "group_extremum":
                return group_extremum(
                    df,
                    params["group_col"],
                    params["target_col"],
                    params.get("agg", "mean"),
                    params.get("extremum", "max"),
                    params.get("return_what", "group"),
                    params.get("filter_expr", ""),
                )
            case "correlation":
                return correlation(
                    df,
                    params["col_a"],
                    params["col_b"],
                    params.get("method", "pearson"),
                    params.get("filter_expr", ""),
                )
            case "count_filter":
                return count_filter(df, params.get("filter_expr", ""))
            case "sort_values":
                return sort_values(
                    df,
                    params["col"],
                    params.get("ascending", True),
                    params.get("top_n", 20),
                )
            case "quantile":
                return quantile(df, params["col"], params.get("q"), params.get("filter_expr", ""))
            case "crosstab":
                return crosstab(
                    df,
                    params["col_a"],
                    params["col_b"],
                    params.get("normalize", ""),
                )
            case "derive_stat":
                return derive_stat(
                    df,
                    params["formula"],
                    params["group_col"],
                    params.get("agg", "mean"),
                    params.get("filter_expr", ""),
                    params.get("group_val"),
                )
            case _:
                return f"Unknown tool: {tool}"
    except KeyError as e:
        return f"Missing required parameter: {e}"
    except Exception as e:
        return f"Error running {tool}: {type(e).__name__}: {e}"


def format_tool_docs() -> str:
    """Generate tool documentation for the system prompt."""
    lines = ["AVAILABLE TOOLS:", ""]
    for name, spec in TOOL_SPECS.items():
        lines.append(f"**{name}**: {spec['desc']}")
        lines.append(f"  Parameters: {spec['params']}")
        lines.append(f"  Example: {json.dumps(spec['example'])}")
        lines.append("")
    return "\n".join(lines)


# ============================================================================
# Helper functions
# ============================================================================

def _coerce_numeric(series: pd.Series, na_values: list[str] = ["?", "", "NA", "N/A"]) -> pd.Series:
    """Coerce object series to numeric, treating specified values as NaN."""
    if series.dtype == "object":
        cleaned = series.replace(na_values, pd.NA)
        return pd.to_numeric(cleaned, errors="coerce")
    return series


# ============================================================================
# Tool implementations
# ============================================================================

def inspect(df: pd.DataFrame, aspect: str, n: int = 5) -> str:
    """Inspect various aspects of the dataframe."""
    match aspect:
        case "head":
            return df.head(n).to_string()
        case "tail":
            return df.tail(n).to_string()
        case "shape":
            return f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
        case "dtypes":
            return df.dtypes.to_string()
        case "info":
            import io
            buf = io.StringIO()
            df.info(buf=buf)
            return buf.getvalue()
        case "columns":
            return "\n".join(df.columns.tolist())
        case "missing":
            missing = df.isnull().sum()
            return missing[missing > 0].to_string() if missing.any() else "No missing values"
        case _:
            return f"Unknown aspect: {aspect}. Use: head, tail, shape, dtypes, info, columns, missing"


def describe(df: pd.DataFrame, include: str = "number") -> str:
    """Statistical summary of the dataframe."""
    include_map = {"number": "number", "object": "object", "all": "all"}
    inc = include_map.get(include, "number")
    return df.describe(include=inc if inc != "number" else None).to_string()


def value_counts(df: pd.DataFrame, col: str, top_n: int = 20) -> str:
    """Get value counts for a column."""
    return df[col].value_counts().head(top_n).to_string()


def unique(df: pd.DataFrame, col: str, top_n: int = 50) -> str:
    """Get unique values in a column."""
    uniques = df[col].dropna().unique()
    total = len(uniques)
    values = uniques[:top_n].tolist()
    result = f"Unique values ({total} total):\n" + "\n".join(str(v) for v in values)
    if total > top_n:
        result += f"\n... and {total - top_n} more"
    return result


def group_stat(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    agg: str = "mean",
    filter_expr: str = "",
    group_val: str | None = None,
) -> str:
    """Calculate aggregated statistic grouped by a column. Auto-coerces '?' to NaN.
    
    If group_val is provided, returns atomic scalar for that group only.
    Otherwise returns all groups (for exploration).
    """
    if filter_expr:
        df = df.query(filter_expr)
    
    # Auto-coerce target column if object dtype
    df = df.copy()
    df[target_col] = _coerce_numeric(df[target_col])
    
    result = df.groupby(group_col)[target_col].agg(agg)
    
    # Atomic mode: return scalar for specific group
    if group_val is not None:
        if group_val not in result.index:
            return f"Group '{group_val}' not found in {group_col}"
        val = result.loc[group_val]
        n = df[df[group_col] == group_val][target_col].count()
        return f"{val:.4f} (n={n})"
    
    # Exploration mode: return all groups
    return result.to_string()


def group_extremum(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    agg: str = "mean",
    extremum: str = "max",
    return_what: str = "group",
    filter_expr: str = "",
) -> str:
    """Find which group has max/min aggregated value. Auto-coerces '?' to NaN.
    
    return_what='group' → returns group name (atomic string)
    return_what='value' → returns the extreme value (atomic number)
    """
    if filter_expr:
        df = df.query(filter_expr)
    
    # Auto-coerce target column if object dtype
    df = df.copy()
    df[target_col] = _coerce_numeric(df[target_col])
    
    grouped = df.groupby(group_col)[target_col].agg(agg)
    if grouped.empty:
        return "No data after filtering"
    
    extremum = extremum.lower()
    if extremum not in {"max", "min"}:
        return "Invalid extremum. Use 'max' or 'min'."
    
    if extremum == "max":
        idx = grouped.idxmax()
        val = grouped.max()
    else:
        idx = grouped.idxmin()
        val = grouped.min()
    
    n = df[df[group_col] == idx][target_col].count()
    
    # Atomic output based on return_what
    if return_what == "value":
        return f"{val:.4f} (n={n})"
    else:  # return_what == "group"
        return f"{idx} ({agg}={val:.4f}, n={n})"


def correlation(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    method: str = "pearson",
    filter_expr: str = "",
) -> str:
    """Calculate correlation coefficient between two columns."""
    if filter_expr:
        df = df.query(filter_expr)
    corr = df[col_a].corr(df[col_b], method=method)
    strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
    direction = "positive" if corr > 0 else "negative"
    return f"{corr:.4f} ({strength} {direction} correlation)"


def count_filter(df: pd.DataFrame, filter_expr: str = "") -> str:
    """Count rows matching a filter expression."""
    total = len(df)
    if filter_expr:
        filtered = df.query(filter_expr)
        count = len(filtered)
        pct = (count / total) * 100 if total > 0 else 0
        return f"{count} rows ({pct:.1f}% of {total} total)"
    return f"{total} rows total"


def sort_values(
    df: pd.DataFrame,
    col: str,
    ascending: bool = True,
    top_n: int = 20,
) -> str:
    """Sort dataframe by a column and show top results."""
    return df.sort_values(col, ascending=ascending).head(top_n).to_string()


def quantile(df: pd.DataFrame, col: str, q: float | list[float] | None = None, filter_expr: str = "") -> str:
    """Calculate quantile(s) for a column.
    
    If q is a single float, returns atomic scalar.
    If q is a list, returns multiple quantiles (for exploration).
    """
    if filter_expr:
        df = df.query(filter_expr)
    
    if q is None:
        q = [0.25, 0.5, 0.75]
    
    # Auto-coerce if needed
    series = _coerce_numeric(df[col])
    
    # Atomic mode: single quantile
    if isinstance(q, (int, float)):
        val = series.quantile(q)
        return f"{val:.4f} (q={q})"
    
    # Exploration mode: multiple quantiles
    result = series.quantile(q)
    return result.to_string()


def crosstab(df: pd.DataFrame, col_a: str, col_b: str, normalize: str = "") -> str:
    """Create a cross-tabulation of two columns."""
    norm = normalize if normalize else False
    ct = pd.crosstab(df[col_a], df[col_b], normalize=norm)
    if normalize:
        ct = ct.round(3)
    return ct.to_string()


def derive_stat(
    df: pd.DataFrame,
    formula: str,
    group_col: str,
    agg: str = "mean",
    filter_expr: str = "",
    group_val: str | None = None,
) -> str:
    """Compute derived metric from formula, aggregate by group.
    
    If group_val is provided, returns atomic scalar for that group only.
    Otherwise returns all groups (for exploration).
    """
    if filter_expr:
        df = df.query(filter_expr)
    
    df = df.copy()
    
    # Auto-coerce columns mentioned in formula that are object dtype
    for col in df.columns:
        if col in formula and df[col].dtype == "object":
            df[col] = _coerce_numeric(df[col])
    
    # Evaluate the formula safely using pd.eval
    try:
        derived = df.eval(formula)
    except Exception as e:
        return f"Error evaluating formula '{formula}': {e}"
    
    df["_derived"] = derived
    
    # Aggregate
    result = df.groupby(group_col)["_derived"].agg(agg)
    
    # Atomic mode: return scalar for specific group
    if group_val is not None:
        if group_val not in result.index:
            return f"Group '{group_val}' not found in {group_col}"
        val = result.loc[group_val]
        n = df[df[group_col] == group_val]["_derived"].count()
        return f"{val:.4f} (n={n})"
    
    # Exploration mode: return all groups
    return f"Formula: {formula}\n\n{result.to_string()}"

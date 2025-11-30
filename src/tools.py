import json
import pandas as pd

# Available tools and their specs for prompt generation
TOOL_SPECS = {
    "inspect": {
        "desc": "View dataframe structure (head, tail, shape, dtypes, columns, missing)",
        "params": {"aspect": "str (required)", "n": "int (default: 5)"},
        "example": {"tool": "inspect", "aspect": "head", "n": 10},
    },
    "describe": {
        "desc": "Statistical summary of columns",
        "params": {"include": "str: 'number', 'object', or 'all' (default: 'number')"},
        "example": {"tool": "describe", "include": "number"},
    },
    "value_counts": {
        "desc": "Frequency counts for a column",
        "params": {"col": "str (required)", "top_n": "int (default: 20)"},
        "example": {"tool": "value_counts", "col": "category"},
    },
    "unique": {
        "desc": "List unique values in a column",
        "params": {"col": "str (required)", "top_n": "int (default: 50)"},
        "example": {"tool": "unique", "col": "status"},
    },
    "group_stat": {
        "desc": "Aggregate a column grouped by another column",
        "params": {
            "group_col": "str (required)",
            "target_col": "str (required)",
            "agg": "str: mean/sum/median/std/min/max/count/nunique (default: 'mean')",
            "filter_expr": "str: pandas query expression (default: '')",
        },
        "example": {"tool": "group_stat", "group_col": "region", "target_col": "sales", "agg": "sum"},
    },
    "group_extremum": {
        "desc": "Find the group(s) with the max or min aggregated value",
        "params": {
            "group_col": "str (required)",
            "target_col": "str (required)",
            "agg": "str: mean/sum/median/std/min/max/count/nunique (default: 'mean')",
            "extremum": "str: 'max' or 'min' (default: 'max')",
            "filter_expr": "str: pandas query expression (default: '')",
        },
        "example": {
            "tool": "group_extremum",
            "group_col": "region",
            "target_col": "sales",
            "agg": "mean",
            "extremum": "max",
        },
    },
    "correlation": {
        "desc": "Correlation between two numeric columns",
        "params": {
            "col_a": "str (required)",
            "col_b": "str (required)",
            "method": "str: pearson/spearman/kendall (default: 'pearson')",
            "filter_expr": "str (default: '')",
        },
        "example": {"tool": "correlation", "col_a": "price", "col_b": "quantity"},
    },
    "count_filter": {
        "desc": "Count rows matching a condition",
        "params": {"filter_expr": "str: pandas query expression (default: '' = all rows)"},
        "example": {"tool": "count_filter", "filter_expr": "age > 30 and status == 'active'"},
    },
    "sort_values": {
        "desc": "Sort by a column and show top rows",
        "params": {
            "col": "str (required)",
            "ascending": "bool (default: true)",
            "top_n": "int (default: 20)",
        },
        "example": {"tool": "sort_values", "col": "score", "ascending": False, "top_n": 10},
    },
    "quantile": {
        "desc": "Calculate percentiles for a column",
        "params": {"col": "str (required)", "q": "list[float] (default: [0.1, 0.25, 0.5, 0.75, 0.9])"},
        "example": {"tool": "quantile", "col": "income"},
    },
    "crosstab": {
        "desc": "Cross-tabulation of two categorical columns",
        "params": {
            "col_a": "str (required)",
            "col_b": "str (required)",
            "normalize": "str: 'index', 'columns', 'all', or '' (default: '')",
        },
        "example": {"tool": "crosstab", "col_a": "gender", "col_b": "purchased", "normalize": "index"},
    },
}


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
                )
            case "group_extremum":
                return group_extremum(
                    df,
                    params["group_col"],
                    params["target_col"],
                    params.get("agg", "mean"),
                    params.get("extremum", "max"),
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
                return quantile(df, params["col"], params.get("q"))
            case "crosstab":
                return crosstab(
                    df,
                    params["col_a"],
                    params["col_b"],
                    params.get("normalize", ""),
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
) -> str:
    """Calculate aggregated statistics grouped by a column."""
    if filter_expr:
        df = df.query(filter_expr)
    return df.groupby(group_col)[target_col].agg(agg).to_string()


def group_extremum(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    agg: str = "mean",
    extremum: str = "max",
    filter_expr: str = "",
) -> str:
    """Find the group(s) with the maximum or minimum aggregated value."""
    if filter_expr:
        df = df.query(filter_expr)
    grouped = df.groupby(group_col)[target_col].agg(agg)
    if grouped.empty:
        return "No data after filtering"
    extremum = extremum.lower()
    if extremum not in {"max", "min"}:
        return "Invalid extremum. Use 'max' or 'min'."
    extreme_value = grouped.max() if extremum == "max" else grouped.min()
    winners = grouped[grouped == extreme_value]
    groups = ", ".join(str(idx) for idx in winners.index)
    label = "Max" if extremum == "max" else "Min"
    return f"{label} {agg} {target_col} by {group_col}: {groups} = {extreme_value}"


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


def quantile(df: pd.DataFrame, col: str, q: list[float] | None = None) -> str:
    """Calculate quantiles/percentiles for a column."""
    if q is None:
        q = [0.1, 0.25, 0.5, 0.75, 0.9]
    result = df[col].quantile(q)
    return result.to_string()


def crosstab(df: pd.DataFrame, col_a: str, col_b: str, normalize: str = "") -> str:
    """Create a cross-tabulation of two columns."""
    norm = normalize if normalize else False
    ct = pd.crosstab(df[col_a], df[col_b], normalize=norm)
    if normalize:
        ct = ct.round(3)
    return ct.to_string()

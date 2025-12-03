import json
import pandas as pd

# Available tools and their specs for prompt generation
# summarize: True = output may be large, should be summarized for context
#            False = output is atomic/small, keep as-is
TOOL_SPECS = {
    "inspect": {
        "name": "inspect",
        "type": "function",
        "description": "View dataframe structure. Use 'columns' first to see available columns, 'shape' for row/col counts, 'head'/'tail' for sample rows, 'dtypes' for column types, 'missing' for null counts, 'info' for memory/dtype summary.",
        "parameters": {
            "type": "object",
            "properties": {
                "aspect": {
                    "type": "string",
                    "description": "The aspect of the dataframe to inspect",
                    "enum": ["head", "tail", "shape", "dtypes", "columns", "missing", "info"]
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
        "description": "Statistical summary (count, mean, std, min, 25%, 50%, 75%, max) for numeric columns. Use include='object' for categorical stats (count, unique, top, freq), or 'all' for both.",
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
        "description": "Frequency counts for a column. Shows how many times each value appears, sorted by frequency (most common first). Good for understanding categorical distributions.",
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
        "description": "List unique values in a column (unsorted, first-seen order). Use value_counts instead if you need frequencies. Good for seeing what distinct values exist.",
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
        "description": "Aggregate a column grouped by another (e.g., mean salary by department). Supports rates via agg=condition_rate|missing_rate. Returns all groups unless group_val is specified for a single group's scalar. Auto-coerces '?' to NaN.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_col": {
                    "type": "string",
                    "description": "Column to group by (e.g., 'department', 'category')"
                },
                "target_col": {
                    "type": "string",
                    "description": "Numeric column to aggregate (e.g., 'salary', 'price') or column to check for missing_rate"
                },
                "agg": {
                    "type": "string",
                    "description": "Aggregation function",
                    "enum": ["mean", "sum", "median", "std", "min", "max", "count", "nunique", "condition_rate", "missing_rate"],
                    "default": "mean"
                },
                "filter_expr": {
                    "type": "string",
                    "description": "Pandas query expression to filter rows before grouping (e.g., \"age > 30\")",
                    "default": ""
                },
                "group_val": {
                    "type": "string",
                    "description": "Return scalar for this specific group only (optional)"
                },
                "condition": {
                    "type": "string",
                    "description": "Condition expression for agg='condition_rate' (e.g., \"TL > 50\")"
                }
            },
            "required": ["group_col"],
            "additionalProperties": False
        },
        "summarize": False
    },
    "multi_group_stat": {
        "name": "multi_group_stat",
        "type": "function",
        "description": "Aggregate multiple target columns with multiple aggs by group. Returns a compact table (one row per group, wide columns like col_agg). Auto-coerces '?' to NaN for numeric aggs.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_col": {
                    "type": "string",
                    "description": "Column to group by"
                },
                "target_cols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of target columns to aggregate"
                },
                "aggs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of aggregations (mean|sum|median|std|min|max|count|nunique)"
                },
                "filter_expr": {
                    "type": "string",
                    "description": "Optional pandas query to filter rows before grouping",
                    "default": ""
                },
                "top_n_cols": {
                    "type": "integer",
                    "description": "Optional limit on number of aggregated columns returned",
                    "default": 50
                }
            },
            "required": ["group_col", "target_cols", "aggs"],
            "additionalProperties": False
        },
        "summarize": True
    },
    "group_extremum": {
        "name": "group_extremum",
        "type": "function",
        "description": "Find which group has the highest/lowest aggregated value (e.g., 'which department has highest mean salary?'). Returns group name by default, or the value with return_what='value'.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_col": {
                    "type": "string",
                    "description": "Column to group by (e.g., 'department')"
                },
                "target_col": {
                    "type": "string",
                    "description": "Numeric column to aggregate (e.g., 'salary')"
                },
                "agg": {
                    "type": "string",
                    "description": "Aggregation function to apply before finding extremum",
                    "enum": ["mean", "sum", "median", "std", "min", "max", "count", "nunique"],
                    "default": "mean"
                },
                "extremum": {
                    "type": "string",
                    "description": "Find the group with maximum or minimum aggregated value",
                    "enum": ["max", "min"],
                    "default": "max"
                },
                "return_what": {
                    "type": "string",
                    "description": "Return the group name or the aggregated value",
                    "enum": ["group", "value"],
                    "default": "group"
                },
                "filter_expr": {
                    "type": "string",
                    "description": "Pandas query expression to filter rows before grouping",
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
        "description": "Correlation coefficient between two numeric columns. Returns value from -1 (inverse) to +1 (direct), with interpretation (weak/moderate/strong). Use spearman for non-linear relationships.",
        "parameters": {
            "type": "object",
            "properties": {
                "col_a": {
                    "type": "string",
                    "description": "First numeric column"
                },
                "col_b": {
                    "type": "string",
                    "description": "Second numeric column"
                },
                "method": {
                    "type": "string",
                    "description": "pearson (linear), spearman (monotonic/ranked), kendall (ordinal)",
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
        "description": "Count rows matching a condition. Returns count and percentage of total. Empty filter_expr returns total row count.",
        "parameters": {
            "type": "object",
            "properties": {
                "filter_expr": {
                    "type": "string",
                    "description": "Pandas query expression (e.g., \"age > 30\", \"status == 'active'\", \"price.between(10, 100)\")",
                    "default": ""
                }
            },
            "required": [],
            "additionalProperties": False
        },
        "summarize": False
    },
    "filter_stat": {
        "name": "filter_stat",
        "type": "function",
        "description": "Aggregate a numeric column after applying an optional filter (no grouping). Always returns a single scalar with count.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_col": {
                    "type": "string",
                    "description": "Numeric column to aggregate (e.g., 'salary', 'price')"
                },
                "agg": {
                    "type": "string",
                    "description": "Aggregation function",
                    "enum": ["mean", "sum", "median", "std", "min", "max", "count", "nunique"],
                    "default": "mean"
                },
                "filter_expr": {
                    "type": "string",
                    "description": "Pandas query expression to filter rows before aggregating",
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
        "description": "Calculate percentile(s) for a column. Use q=0.5 for median, q=0.9 for 90th percentile, etc. Single q returns scalar; list returns multiple.",
        "parameters": {
            "type": "object",
            "properties": {
                "col": {
                    "type": "string",
                    "description": "Numeric column name"
                },
                "q": {
                    "oneOf": [
                        {"type": "number", "description": "Single quantile 0-1 (e.g., 0.5 for median)"},
                        {"type": "array", "items": {"type": "number"}, "description": "List of quantiles"}
                    ],
                    "description": "Quantile value(s) between 0 and 1",
                    "default": [0.25, 0.5, 0.75]
                },
                "filter_expr": {
                    "type": "string",
                    "description": "Pandas query expression to filter rows",
                    "default": ""
                }
            },
            "required": ["col"],
            "additionalProperties": False
        },
        "summarize": False
    },
    "crosstab": {
        "name": "crosstab",
        "type": "function",
        "description": "Cross-tabulation (contingency table) of two categorical columns. Shows counts at each intersection. Use normalize for proportions.",
        "parameters": {
            "type": "object",
            "properties": {
                "col_a": {
                    "type": "string",
                    "description": "Row categories (first column)"
                },
                "col_b": {
                    "type": "string",
                    "description": "Column categories (second column)"
                },
                "normalize": {
                    "type": "string",
                    "description": "'' for counts, 'index' for row %, 'columns' for col %, 'all' for total %",
                    "enum": ["index", "columns", "all", ""],
                    "default": ""
                }
            },
            "required": ["col_a", "col_b"],
            "additionalProperties": False
        },
        "summarize": True
    },
    "group_value_counts": {
        "name": "group_value_counts",
        "type": "function",
        "description": "Top-N value counts per group for a categorical target. Returns stacked table: group | value | count | pct.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_col": {
                    "type": "string",
                    "description": "Column to group by (e.g., 'TR')"
                },
                "target_col": {
                    "type": "string",
                    "description": "Categorical column to count values of (e.g., 'IN')"
                },
                "top_n": {
                    "type": "integer",
                    "description": "Top N values per group",
                    "default": 5
                },
                "normalize": {
                    "type": "boolean",
                    "description": "If true, include percentage of group total",
                    "default": True
                }
            },
            "required": ["group_col", "target_col"],
            "additionalProperties": False
        },
        "summarize": True
    },
    "missing_rate": {
        "name": "missing_rate",
        "type": "function",
        "description": "Compute missingness for a column (optionally within a group). Returns percent missing and count.",
        "parameters": {
            "type": "object",
            "properties": {
                "col": {
                    "type": "string",
                    "description": "Column to check for missing values"
                },
                "group_col": {
                    "type": "string",
                    "description": "Optional column to group by before computing missingness"
                },
                "group_val": {
                    "type": "string",
                    "description": "Return scalar for this specific group (required if group_col provided)"
                }
            },
            "required": ["col"],
            "additionalProperties": False
        },
        "summarize": False
    },
    "group_condition_rate": {
        "name": "group_condition_rate",
        "type": "function",
        "description": "Within a group, compute the percentage of rows satisfying a condition. Requires group_val to keep output atomic.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_col": {
                    "type": "string",
                    "description": "Column to group by (e.g., 'department')"
                },
                "condition": {
                    "type": "string",
                    "description": "Pandas query expression for rows counted as true (e.g., \"TL > 50\")"
                },
                "group_val": {
                    "type": "string",
                    "description": "Return scalar for this specific group only"
                }
            },
            "required": ["group_col", "condition", "group_val"],
            "additionalProperties": False
        },
        "summarize": False
    },
    "derive_stat": {
        "name": "derive_stat",
        "type": "function",
        "description": "Compute a derived metric from a formula (e.g., 'revenue / cost', 'A + B * 2'), then aggregate by group. Useful for ratios, differences, or computed columns.",
        "parameters": {
            "type": "object",
            "properties": {
                "formula": {
                    "type": "string",
                    "description": "Python expression using column names (e.g., 'revenue / cost', 'price * quantity')"
                },
                "group_col": {
                    "type": "string",
                    "description": "Column to group by for aggregation"
                },
                "agg": {
                    "type": "string",
                    "description": "Aggregation function to apply to computed values",
                    "enum": ["mean", "sum", "median", "std", "min", "max", "count"],
                    "default": "mean"
                },
                "filter_expr": {
                    "type": "string",
                    "description": "Pandas query expression to filter rows before computation",
                    "default": ""
                },
                "group_val": {
                    "type": "string",
                    "description": "Return scalar for this specific group only (optional)"
                }
            },
            "required": ["formula", "group_col"],
            "additionalProperties": False
        },
        "summarize": False
    },
    # ========================================================================
    # Hook-chaining tools: operate on previous hook results, not raw dataframe
    # ========================================================================
    "combine": {
        "name": "combine",
        "type": "function",
        "description": "Compute arithmetic or boolean expressions on previous hook results. Use for ratios, differences, percentages, comparisons. Returns number or 'true'/'false'.",
        "parameters": {
            "type": "object",
            "properties": {
                "expr": {
                    "type": "string",
                    "description": "Expression using variable names (e.g., 'a / b', '(a - b) / b * 100', 'a > b', 'a == b')"
                },
                "vars": {
                    "type": "object",
                    "description": "Mapping of variable names to hook IDs (e.g., {'a': 'hook1', 'b': 'hook2'})",
                    "additionalProperties": {"type": "string"}
                }
            },
            "required": ["expr", "vars"],
            "additionalProperties": False
        },
        "summarize": False,
        "chain": True  # Marks this as a hook-chaining tool
    },
    "lookup": {
        "name": "lookup",
        "type": "function",
        "description": "Use a hook's group result to query another stat. Chains group_extremum → group_stat dynamically without hardcoding group names.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_hook": {
                    "type": "string",
                    "description": "Hook ID that returned a group name (e.g., from group_extremum)"
                },
                "group_col": {
                    "type": "string",
                    "description": "Column the group belongs to"
                },
                "target_col": {
                    "type": "string",
                    "description": "Column to aggregate for the looked-up group"
                },
                "agg": {
                    "type": "string",
                    "description": "Aggregation function",
                    "enum": ["mean", "sum", "median", "std", "min", "max", "count", "nunique"],
                    "default": "mean"
                }
            },
            "required": ["group_hook", "group_col", "target_col"],
            "additionalProperties": False
        },
        "summarize": False,
        "chain": True
    },
    "aggregate_hooks": {
        "name": "aggregate_hooks",
        "type": "function",
        "description": "Compute min/max/mean/sum/range across multiple hook values. Useful for finding extremes or averages across a set of computed values.",
        "parameters": {
            "type": "object",
            "properties": {
                "hooks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of hook IDs to aggregate"
                },
                "agg": {
                    "type": "string",
                    "description": "Aggregation function",
                    "enum": ["min", "max", "mean", "sum", "range"]
                }
            },
            "required": ["hooks", "agg"],
            "additionalProperties": False
        },
        "summarize": False,
        "chain": True
    }
}


def is_chain_tool(tool_name: str) -> bool:
    """Check if a tool operates on hook results rather than the dataframe."""
    return TOOL_SPECS.get(tool_name, {}).get("chain", False)


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
                    params.get("target_col"),
                    params.get("agg", "mean"),
                    params.get("filter_expr", ""),
                    params.get("group_val"),
                    params.get("condition"),
                )
            case "multi_group_stat":
                return multi_group_stat(
                    df,
                    params["group_col"],
                    params["target_cols"],
                    params["aggs"],
                    params.get("filter_expr", ""),
                    params.get("top_n_cols", 50),
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
            case "filter_stat":
                return filter_stat(
                    df,
                    params["target_col"],
                    params.get("agg", "mean"),
                    params.get("filter_expr", ""),
                )
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
            case "group_value_counts":
                return group_value_counts(
                    df,
                    params["group_col"],
                    params["target_col"],
                    params.get("top_n", 5),
                    params.get("normalize", True),
                )
            case "missing_rate":
                return missing_rate(
                    df,
                    params["col"],
                    params.get("group_col"),
                    params.get("group_val"),
                )
            case "group_condition_rate":
                return group_condition_rate(
                    df,
                    params["group_col"],
                    params["condition"],
                    params["group_val"],
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
        lines.append(f"**{name}**: {spec['description']}")
        # Format parameters
        params = spec.get("parameters", {}).get("properties", {})
        required = spec.get("parameters", {}).get("required", [])
        param_strs = []
        for pname, pinfo in params.items():
            req = "*" if pname in required else ""
            ptype = pinfo.get("type", pinfo.get("oneOf", [{}])[0].get("type", "any"))
            default = f" (default: {pinfo['default']})" if "default" in pinfo else ""
            param_strs.append(f"{pname}{req}: {ptype}{default}")
        if param_strs:
            lines.append(f"  Parameters: {', '.join(param_strs)}")
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


def _coerce_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a copy of df with specified columns coerced to numeric."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = _coerce_numeric(df[col])
    return df


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
    if include == "number":
        return df.describe().to_string()  # pandas default: numeric only
    elif include == "object":
        return df.describe(include="object").to_string()
    else:  # "all"
        return df.describe(include="all").to_string()


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
    target_col: str | None,
    agg: str = "mean",
    filter_expr: str = "",
    group_val: str | None = None,
    condition: str | None = None,
) -> str:
    """Calculate aggregated statistic grouped by a column. Auto-coerces '?' to NaN.
    
    Supports rates:
      - agg='condition_rate' with condition expression
      - agg='missing_rate' on target_col
    
    If group_val is provided, returns atomic scalar for that group only.
    Otherwise returns all groups (for exploration).
    """
    if filter_expr:
        df = df.query(filter_expr)
    
    # Rates branch
    if agg == "condition_rate":
        if condition is None or condition == "":
            return "condition is required for agg='condition_rate'"
        totals = df.groupby(group_col).size()
        satisfied = df.query(condition).groupby(group_col).size()
        rates = (satisfied / totals * 100).fillna(0)
        if group_val is not None:
            if group_val not in totals.index:
                return f"Group '{group_val}' not found in {group_col}"
            pct = rates.get(group_val, 0)
            count = satisfied.get(group_val, 0)
            total = totals.get(group_val, 0)
            return f"{pct:.2f}% ({count} of {total})"
        return rates.to_string()
    
    if agg == "missing_rate":
        if target_col is None:
            return "target_col is required for agg='missing_rate'"
        # Normalize missing markers
        df = df.copy()
        df[target_col] = df[target_col].replace(["?", "", "NA", "N/A"], pd.NA)
        totals = df.groupby(group_col).size()
        missing = df[target_col].isna().groupby(df[group_col]).sum()
        rates = (missing / totals * 100).fillna(0)
        if group_val is not None:
            if group_val not in totals.index:
                return f"Group '{group_val}' not found in {group_col}"
            pct = rates.get(group_val, 0)
            miss_ct = missing.get(group_val, 0)
            total = totals.get(group_val, 0)
            return f"{pct:.2f}% missing ({miss_ct} of {total})"
        return rates.to_string()
    
    # Standard aggregations
    if target_col is None:
        return "target_col is required for this aggregation"
    
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
    
    ext = extremum.lower()
    if ext not in {"max", "min"}:
        return "Invalid extremum. Use 'max' or 'min'."
    
    if ext == "max":
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
    df = _coerce_numeric_columns(df, [col_a, col_b])
    corr = df[col_a].corr(df[col_b], method=method)
    strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
    direction = "positive" if corr > 0 else "negative"
    return f"{corr:.4f} ({strength} {direction} correlation)"


def count_filter(df: pd.DataFrame, filter_expr: str = "") -> str:
    """Count rows matching a filter expression."""
    return filter_stat(df, target_col=None, agg="count", filter_expr=filter_expr)


def filter_stat(df: pd.DataFrame, target_col: str | None, agg: str = "mean", filter_expr: str = "") -> str:
    """Aggregate a numeric column with optional filtering (no grouping)."""
    total_rows = len(df)
    if filter_expr:
        df = df.query(filter_expr)
    if df.empty:
        return "No data after filtering"
    if agg == "count":
        count = len(df)
        pct = (count / total_rows * 100) if total_rows > 0 else 0
        return f"{count} rows ({pct:.1f}% of {total_rows} total)"
    if target_col is None:
        return "target_col is required for this aggregation"
    df = _coerce_numeric_columns(df, [target_col])
    series = df[target_col]
    result = series.agg(agg)
    n = series.count()
    return f"{result:.4f} (n={n})"


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


def missing_rate(df: pd.DataFrame, col: str, group_col: str | None = None, group_val: str | None = None) -> str:
    """Compute missingness percent for a column, optionally within a group."""
    if group_col:
        if group_val is None:
            return "group_val required when group_col is provided"
        df = df[df[group_col] == group_val]
    total = len(df)
    if total == 0:
        return "No data after filtering"
    missing = df[col].replace(["?", "", "NA", "N/A"], pd.NA).isna().sum()
    pct = (missing / total) * 100
    return f"{pct:.2f}% missing ({missing} of {total})"


def group_condition_rate(df: pd.DataFrame, group_col: str, condition: str, group_val: str) -> str:
    """Within a group, compute % of rows satisfying condition."""
    subset = df[df[group_col] == group_val]
    total = len(subset)
    if total == 0:
        return f"Group '{group_val}' not found in {group_col}"
    satisfied = subset.query(condition)
    count = len(satisfied)
    pct = (count / total) * 100
    return f"{pct:.2f}% ({count} of {total})"


def multi_group_stat(
    df: pd.DataFrame,
    group_col: str,
    target_cols: list[str],
    aggs: list[str],
    filter_expr: str = "",
    top_n_cols: int = 50,
) -> str:
    """Aggregate multiple targets/aggregations by group; return compact table."""
    if filter_expr:
        df = df.query(filter_expr)
    if df.empty:
        return "No data after filtering"
    
    df = df.copy()
    # Coerce numeric-like targets
    for col in target_cols:
        df[col] = _coerce_numeric(df[col])
    
    grouped = df.groupby(group_col)[target_cols].agg(aggs)
    # Flatten MultiIndex columns
    grouped.columns = [f"{col}_{agg}" for col, agg in grouped.columns]
    
    # Limit column count to keep output compact
    if len(grouped.columns) > top_n_cols:
        grouped = grouped[grouped.columns[:top_n_cols]]
    
    return grouped.reset_index().to_string(index=False)


def group_value_counts(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    top_n: int = 5,
    normalize: bool = True,
) -> str:
    """Top-N value counts per group for a categorical column."""
    results = []
    for group, sub in df.groupby(group_col):
        vc = sub[target_col].value_counts().head(top_n)
        total = len(sub)
        for value, count in vc.items():
            pct = (count / total * 100) if normalize and total > 0 else None
            if normalize:
                results.append((group, value, count, f"{pct:.1f}%"))
            else:
                results.append((group, value, count, ""))
    if not results:
        return "No data after filtering"
    
    # Build a small table
    header = ["group", "value", "count"] + (["pct"] if normalize else [])
    lines = [" | ".join(header)]
    for row in results:
        if normalize:
            g, v, c, pct = row
            lines.append(f"{g} | {v} | {c} | {pct}")
        else:
            g, v, c, _ = row
            lines.append(f"{g} | {v} | {c}")
    return "\n".join(lines)


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


# ============================================================================
# Hook-chaining tool implementations
# These operate on previous hook results, not the raw dataframe
# ============================================================================

def _parse_hook_value(output: str) -> float:
    """
    Parse numeric value from hook output string.
    Handles formats like:
      - "45.2300 (n=45)"
      - "control (mean=45.23, n=45)"
      - "0.8234 (strong positive correlation)"
      - "123 rows (45.2% of 273 total)"
      - plain numbers
    """
    import re
    
    # Try plain float first
    try:
        return float(output.strip())
    except ValueError:
        pass
    
    # Pattern: number at start followed by space/paren
    match = re.match(r'^(-?\d+\.?\d*)', output.strip())
    if match:
        return float(match.group(1))
    
    # Pattern: "group (mean=X, ...)" - extract first number after =
    match = re.search(r'=(-?\d+\.?\d*)', output)
    if match:
        return float(match.group(1))
    
    raise ValueError(f"Cannot parse numeric value from: {output}")


def _parse_hook_group(output: str) -> str:
    """
    Parse group name from hook output string.
    Handles format: "group_name (mean=X, n=Y)"
    """
    import re
    match = re.match(r'^([^\(]+)', output.strip())
    if match:
        return match.group(1).strip()
    return output.strip()


def combine(hook_results: dict[str, str], expr: str, vars: dict[str, str]) -> str:
    """
    Compute arithmetic or boolean expression on hook results.
    
    Args:
        hook_results: Dict mapping hook_id -> output string
        expr: Expression using variable names (arithmetic or comparison)
        vars: Dict mapping variable names -> hook IDs
    
    Returns:
        Computed result as string (number or "true"/"false")
    """
    # Build namespace with parsed values
    namespace = {}
    for var_name, hook_id in vars.items():
        if hook_id not in hook_results:
            return f"Hook '{hook_id}' not found in results"
        try:
            namespace[var_name] = _parse_hook_value(hook_results[hook_id])
        except ValueError as e:
            return f"Error parsing hook '{hook_id}': {e}"
    
    # Evaluate expression safely
    try:
        # Use restricted eval - no builtins, only our namespace
        result = eval(expr, {"__builtins__": {}}, namespace)
        
        # Handle boolean results
        if isinstance(result, bool):
            return "true" if result else "false"
        
        # Handle numeric results
        return f"{result:.4f}"
    except Exception as e:
        return f"Error evaluating '{expr}': {e}"


def lookup(
    df: pd.DataFrame,
    hook_results: dict[str, str],
    group_hook: str,
    group_col: str,
    target_col: str,
    agg: str = "mean",
) -> str:
    """
    Use a hook's group result to query another stat.
    
    Args:
        df: The dataframe
        hook_results: Dict mapping hook_id -> output string
        group_hook: Hook ID that returned a group name
        group_col: Column the group belongs to
        target_col: Column to aggregate
        agg: Aggregation function
    
    Returns:
        Aggregated stat for the looked-up group
    """
    if group_hook not in hook_results:
        return f"Hook '{group_hook}' not found in results"
    
    try:
        group_name = _parse_hook_group(hook_results[group_hook])
    except Exception as e:
        return f"Error parsing group from hook '{group_hook}': {e}"
    
    # Use group_stat logic
    return group_stat(df, group_col, target_col, agg, filter_expr="", group_val=group_name)


def aggregate_hooks(hook_results: dict[str, str], hooks: list[str], agg: str) -> str:
    """
    Aggregate values across multiple hooks.
    
    Args:
        hook_results: Dict mapping hook_id -> output string
        hooks: List of hook IDs to aggregate
        agg: Aggregation (min, max, mean, sum, range)
    
    Returns:
        Aggregated result
    """
    values = []
    for hook_id in hooks:
        if hook_id not in hook_results:
            return f"Hook '{hook_id}' not found in results"
        try:
            values.append(_parse_hook_value(hook_results[hook_id]))
        except ValueError as e:
            return f"Error parsing hook '{hook_id}': {e}"
    
    if not values:
        return "No values to aggregate"
    
    match agg:
        case "min":
            result = min(values)
            idx = hooks[values.index(result)]
            return f"{result:.4f} (from {idx})"
        case "max":
            result = max(values)
            idx = hooks[values.index(result)]
            return f"{result:.4f} (from {idx})"
        case "mean":
            result = sum(values) / len(values)
            return f"{result:.4f} (n={len(values)})"
        case "sum":
            result = sum(values)
            return f"{result:.4f} (n={len(values)})"
        case "range":
            result = max(values) - min(values)
            return f"{result:.4f} (max={max(values):.4f}, min={min(values):.4f})"
        case _:
            return f"Unknown aggregation: {agg}"


def run_chain_tool(
    tool: str,
    df: pd.DataFrame,
    hook_results: dict[str, str],
    params: dict,
) -> str:
    """
    Run a hook-chaining tool.
    
    Args:
        tool: Tool name
        df: Dataframe (only needed for 'lookup')
        hook_results: Dict mapping hook_id -> output string
        params: Tool parameters
    
    Returns:
        Tool output string
    """
    try:
        match tool:
            case "combine":
                return combine(hook_results, params["expr"], params["vars"])
            case "lookup":
                return lookup(
                    df,
                    hook_results,
                    params["group_hook"],
                    params["group_col"],
                    params["target_col"],
                    params.get("agg", "mean"),
                )
            case "aggregate_hooks":
                return aggregate_hooks(hook_results, params["hooks"], params["agg"])
            case _:
                return f"Unknown chain tool: {tool}"
    except KeyError as e:
        return f"Missing required parameter: {e}"
    except Exception as e:
        return f"Error running {tool}: {type(e).__name__}: {e}"

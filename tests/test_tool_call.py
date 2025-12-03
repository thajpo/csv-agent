import pandas as pd
from src.tools import parse_tool_call, run_tool


def test_parse_valid_tool_call():
    """Test parsing a valid JSON tool call."""
    json_str = '{"tool": "inspect", "aspect": "shape"}'
    result = parse_tool_call(json_str)
    
    assert isinstance(result, tuple)
    tool_name, params = result
    assert tool_name == "inspect"
    assert params == {"aspect": "shape"}


def test_parse_tool_call_with_params():
    """Test parsing a tool call with multiple parameters."""
    json_str = '{"tool": "value_counts", "col": "category", "top_n": 10}'
    result = parse_tool_call(json_str)
    
    assert isinstance(result, tuple)
    tool_name, params = result
    assert tool_name == "value_counts"
    assert params == {"col": "category", "top_n": 10}


def test_parse_invalid_json():
    """Test parsing invalid JSON returns error string."""
    json_str = '{"tool": "inspect", "aspect": "shape"'  # missing closing brace
    result = parse_tool_call(json_str)
    
    assert isinstance(result, str)
    assert "Invalid JSON" in result


def test_parse_missing_tool_field():
    """Test parsing JSON without 'tool' field returns error."""
    json_str = '{"aspect": "shape"}'
    result = parse_tool_call(json_str)
    
    assert isinstance(result, str)
    assert "Missing 'tool' field" in result


def test_parse_unknown_tool():
    """Test parsing unknown tool returns error."""
    json_str = '{"tool": "nonexistent_tool", "param": "value"}'
    result = parse_tool_call(json_str)
    
    assert isinstance(result, str)
    assert "Unknown tool" in result


def test_run_tool_inspect_shape():
    """Test running the inspect tool for shape."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    result = run_tool("inspect", df, {"aspect": "shape"})

    print(result)
    
    assert isinstance(result, str)
    assert "Rows: 3" in result
    assert "Columns: 2" in result


def test_run_tool_inspect_head():
    """Test running the inspect tool for head."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    result = run_tool("inspect", df, {"aspect": "head", "n": 2})
    
    assert isinstance(result, str)
    assert "A" in result
    assert "B" in result


def test_run_tool_count_filter():
    """Test running count_filter tool."""
    df = pd.DataFrame({"age": [25, 30, 35, 40], "status": ["A", "B", "A", "B"]})
    result = run_tool("count_filter", df, {"filter_expr": "age > 30"})
    
    assert isinstance(result, str)
    assert "2 rows" in result


def test_run_tool_value_counts():
    """Test running value_counts tool."""
    df = pd.DataFrame({"category": ["A", "B", "A", "B", "A"]})
    result = run_tool("value_counts", df, {"col": "category"})
    
    assert isinstance(result, str)
    assert "A" in result
    assert "B" in result


def test_run_tool_missing_required_param():
    """Test running tool with missing required parameter."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    result = run_tool("value_counts", df, {})  # missing "col"
    
    assert isinstance(result, str)
    assert "Missing required parameter" in result


def test_end_to_end_parse_and_run():
    """Test full flow: parse tool call and run it."""
    # Parse
    json_str = '{"tool": "inspect", "aspect": "columns"}'
    parse_result = parse_tool_call(json_str)
    
    assert isinstance(parse_result, tuple)
    tool_name, params = parse_result
    
    # Run
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    result = run_tool(tool_name, df, params)
    
    assert isinstance(result, str)
    assert "col1" in result
    assert "col2" in result


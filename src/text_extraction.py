"""Text extraction utilities for parsing LLM responses."""
import json
import re


def extract_code_blocks(text: str) -> list[str]:
    """Extract all code between <code> and </code> tags."""
    pattern = r'<code>(.*?)</code>'
    return re.findall(pattern, text, re.DOTALL)


def extract_json_episodes(text: str) -> list[dict] | None:
    """
    Extract JSON episode array from the final output.
    Handles: ```json blocks, raw arrays, and truncated output.
    Returns only valid episode dicts (must have 'question_text' key).
    """
    return _extract_structured_array(text, _is_valid_episode)


def extract_question_plans(text: str) -> list[dict] | None:
    """
    Extract question blueprints (question_text + reasoning_path) from model output.
    Accepts ```json blocks, raw arrays, or truncated arrays.
    """
    return _extract_structured_array(text, _is_valid_question_plan)


def extract_questions(text: str) -> list[str]:
    """Extract all questions between {question} and {/question} tags."""
    pattern = r'\{question\}(.*?)\{/question\}'
    return [q.strip() for q in re.findall(pattern, text, re.DOTALL)]


def extract_json_array(text: str) -> list[dict] | None:
    """
    Extract any JSON array from text.
    Returns list of dicts if found, None otherwise.
    """
    # Try ```json blocks first (most reliable)
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    matches = re.findall(json_pattern, text)
    
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    
    # Try raw array - look for array start pattern (handles nested [ in earlier text)
    # Search for [ followed by newline and { (typical JSON array of objects)
    array_starts = [m.start() for m in re.finditer(r'\[\s*\n?\s*\{', text)]
    
    for start in reversed(array_starts):
        candidate = _extract_balanced_brackets(text, start)
        if candidate:
            try:
                data = json.loads(candidate)
                if isinstance(data, list) and len(data) > 0:
                    return data
            except json.JSONDecodeError:
                pass
    
    return None

def _is_valid_episode(obj) -> bool:
    """Check if object looks like a valid episode."""
    return (
        isinstance(obj, dict) 
        and 'question_text' in obj 
        and 'hooks' in obj
    )


def _is_valid_question_plan(obj) -> bool:
    """Check if object looks like a valid question plan."""
    return (
        isinstance(obj, dict)
        and 'question_text' in obj
        and 'reasoning_path' in obj
    )


def _try_parse_array(text: str, validator) -> list[dict] | None:
    """Try to parse text as a JSON array of objects that pass validator."""
    try:
        data = json.loads(text)
        if isinstance(data, list) and len(data) > 0:
            objects = [obj for obj in data if validator(obj)]
            return objects if objects else None
    except json.JSONDecodeError:
        pass
    return None


def _extract_structured_array(text: str, validator) -> list[dict] | None:
    """Shared extractor for validated JSON arrays (with truncated handling)."""
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    matches = re.findall(json_pattern, text)
    
    for match in matches:
        objects = _try_parse_array(match.strip(), validator)
        if objects:
            return objects
    
    array_starts = [m.start() for m in re.finditer(r'\[\s*\n?\s*\{', text)]
    for start in reversed(array_starts):
        candidate = _extract_balanced_brackets(text, start)
        if candidate:
            objects = _try_parse_array(candidate, validator)
            if objects:
                return objects
        objects = _extract_partial_objects(text[start:], validator)
        if objects:
            return objects
    
    return None


def _extract_partial_episodes(text: str) -> list[dict] | None:
    """
    Extract individual episode objects from potentially truncated JSON.
    Finds complete {...} objects that look like valid episodes.
    """
    return _extract_partial_objects(text, _is_valid_episode)


def _extract_balanced_braces(text: str, start: int) -> str | None:
    """Extract a balanced {...} substring starting at position start."""
    if start >= len(text) or text[start] != '{':
        return None
    
    brace_count = 0
    in_string = False
    escape = False
    
    for i in range(start, len(text)):
        char = text[i]
        
        if escape:
            escape = False
            continue
        
        if char == '\\':
            escape = True
            continue
        
        if char == '"':
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start:i + 1]
    
    return None  # Unbalanced


def _extract_balanced_brackets(text: str, start: int) -> str | None:
    """Extract a balanced [...] substring starting at position start."""
    if start >= len(text) or text[start] != '[':
        return None
    
    bracket_count = 0
    in_string = False
    escape = False
    
    for i in range(start, len(text)):
        char = text[i]
        
        if escape:
            escape = False
            continue
        
        if char == '\\':
            escape = True
            continue
        
        if char == '"':
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if bracket_count == 0:
                return text[start:i + 1]
    
    return None  # Unbalanced


def _extract_partial_objects(text: str, validator) -> list[dict] | None:
    """Extract balanced JSON objects from text that satisfy validator."""
    objects = []
    i = 0
    
    while i < len(text):
        if text[i] == '{':
            obj_str = _extract_balanced_braces(text, i)
            if obj_str:
                try:
                    obj = json.loads(obj_str)
                    if validator(obj):
                        objects.append(obj)
                except json.JSONDecodeError:
                    pass
                i += len(obj_str)
            else:
                i += 1
        else:
            i += 1
    
    return objects if objects else None

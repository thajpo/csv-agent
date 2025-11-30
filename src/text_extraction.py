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
    # Try 1: Find ```json ... ``` block
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    matches = re.findall(json_pattern, text)
    
    for match in matches:
        episodes = _try_parse_array(match.strip())
        if episodes:
            return episodes
    
    # Try 2: Find raw JSON array starting with [ { (avoid earlier [ in examples)
    array_starts = [m.start() for m in re.finditer(r'\[\s*\n?\s*\{', text)]
    for start in reversed(array_starts):
        candidate = _extract_balanced_brackets(text, start)
        if candidate:
            episodes = _try_parse_array(candidate)
            if episodes:
                return episodes
        # Try 3: Array was truncated - extract individual objects from this point
        episodes = _extract_partial_episodes(text[start:])
        if episodes:
            return episodes
    
    return None

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


def _try_parse_array(text: str) -> list[dict] | None:
    """Try to parse text as a JSON array of episodes."""
    try:
        data = json.loads(text)
        if isinstance(data, list) and len(data) > 0:
            # Filter to only valid episodes
            episodes = [ep for ep in data if _is_valid_episode(ep)]
            return episodes if episodes else None
    except json.JSONDecodeError:
        pass
    return None


def _extract_partial_episodes(text: str) -> list[dict] | None:
    """
    Extract individual episode objects from potentially truncated JSON.
    Finds complete {...} objects that look like valid episodes.
    """
    episodes = []
    
    # Find balanced braces starting with {"
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Try to extract a balanced object
            obj_str = _extract_balanced_braces(text, i)
            if obj_str:
                try:
                    obj = json.loads(obj_str)
                    if _is_valid_episode(obj):
                        episodes.append(obj)
                except json.JSONDecodeError:
                    pass
                i += len(obj_str)
            else:
                i += 1
        else:
            i += 1
    
    return episodes if episodes else None


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

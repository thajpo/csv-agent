"""
LLM-based verbalization of code compositions into natural language questions.

Converts deterministic code templates into questions that:
1. Describe WHAT to find, not HOW
2. Reference column PROPERTIES, not column NAMES
3. Include hints that guide without giving away the solution
"""

import json
import re
from typing import Any

from src.core.model import APILLM


VERBALIZATION_PROMPT_K = '''You are generating diverse natural language questions for a data science training dataset.

## DATASET CONTEXT:
{dataset_description}

## DATA OVERVIEW:
{data_overview}

## CODE THAT PRODUCES THE ANSWER:
```python
{code}
```

## GROUND TRUTH ANSWER:
{ground_truth}

## REQUIRED OUTPUT FORMAT:
{output_schema}

## YOUR TASK:
Generate {k} DIVERSE question phrasings that would lead someone to discover this answer through data exploration.

Think: "What question would a data scientist naturally ask that leads to this analysis?"

QUESTION STYLES (use a mix):
1. **Hypothesis-driven**: "I wonder if..." / "I suspect that..." / "Could it be that..."
2. **Direct**: Clear, straightforward ask
3. **Contextual**: Uses domain knowledge (e.g., "For patient outcomes..." if health data)
4. **Exploratory**: "Before modeling, I want to understand..." / "What patterns exist in..."

CRITICAL RULES:
1. DO NOT mention specific column names - reference by properties ("the most variable column")
2. Each question MUST include the required output format
3. Each question should feel natural, not mechanical
4. The hint should guide exploration without revealing the solution
5. Make questions genuinely DIFFERENT from each other (not just rewording)

OUTPUT FORMAT (respond with ONLY this JSON array, no other text):
[
  {{"question": "First question phrasing here...", "hint": "Hint for first question"}},
  {{"question": "Second question phrasing here...", "hint": "Hint for second question"}},
  ...
]'''


VERBALIZATION_PROMPT = '''You are converting a data analysis code snippet into a natural language question for a data science training dataset.

## CODE BEING EXECUTED:
```python
{code}
```

## DATASET INFO:
- Rows: {n_rows}
- Columns: {n_cols}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}

## GROUND TRUTH ANSWER:
{ground_truth}

## REQUIRED OUTPUT FORMAT:
{output_schema}

## YOUR TASK:
Generate a question that describes WHAT to find, WITHOUT revealing HOW to find it.

CRITICAL RULES:
1. DO NOT mention specific column names in the question
2. Reference columns by their PROPERTIES (e.g., "the column with highest variance", "numeric columns", "the most correlated pair")
3. The question should require EXPLORATION to answer - the solver must discover which columns to use
4. The hint should guide the approach without giving away the solution
5. **IMPORTANT**: The question MUST specify the exact output format from the REQUIRED OUTPUT FORMAT section above. Include this as part of the question itself.

GOOD EXAMPLES:
- "What is the mean of the numeric column with the highest variance? Provide your answer as a single number rounded to 3 decimal places."
- "Which pair of numeric columns has the strongest correlation? Provide your answer as a JSON object with keys 'columns' (list of two column names, alphabetically sorted) and 'correlation' (rounded to 3 decimal places)."
- "How many columns have more than 5% missing values? Provide your answer as a JSON object with keys 'count' (integer) and 'columns' (list of column names, alphabetically sorted)."

BAD EXAMPLES (DO NOT DO THIS):
- "What is the mean of the 'alcohol' column?" (names specific column)
- "Calculate df.var().idxmax()" (reveals the code)
- "The answer is 10.5" (gives away the answer)
- "What is the correlation?" (doesn't specify output format)

OUTPUT FORMAT (respond with ONLY this JSON, no other text):
{{
    "question": "Your natural language question here, INCLUDING the required answer format",
    "hint": "A brief hint that guides exploration without revealing the solution"
}}'''


class QuestionVerbalizer:
    """Convert code compositions into natural language questions using an LLM."""

    def __init__(self, model: str, sampling_args: dict):
        """
        Initialize the verbalizer.

        Args:
            model: Model identifier (e.g., "openai/gpt-oss-120b")
            sampling_args: Sampling configuration dict
        """
        self.llm = APILLM(model=model, sampling_args=sampling_args)

    async def verbalize(
        self,
        code: str,
        profile: dict,
        ground_truth: Any,
        output_schema: str = "",
    ) -> tuple[str, str]:
        """
        Convert code into a natural language question.

        Args:
            code: The Python code that was executed
            profile: Dataset profile from DataProfiler
            ground_truth: The actual answer from executing the code
            output_schema: Description of the exact expected output format

        Returns:
            Tuple of (question_text, hint)
        """
        # Extract column info from profile
        columns = profile.get("columns", {})
        numeric_cols = [k for k, v in columns.items() if v.get("type") == "numeric"]
        categorical_cols = [k for k, v in columns.items() if v.get("type") == "categorical"]

        # Format ground truth for display
        if isinstance(ground_truth, dict):
            gt_str = json.dumps(ground_truth, indent=2)
        else:
            gt_str = str(ground_truth)

        # Build prompt
        prompt = VERBALIZATION_PROMPT.format(
            code=code.strip(),
            n_rows=profile.get("shape", {}).get("rows", "unknown"),
            n_cols=profile.get("shape", {}).get("columns", "unknown"),
            numeric_cols=", ".join(numeric_cols[:10]) if numeric_cols else "none",
            categorical_cols=", ".join(categorical_cols[:10]) if categorical_cols else "none",
            ground_truth=gt_str,
            output_schema=output_schema or "Not specified",
        )

        # Call LLM
        response = await self.llm(prompt)

        # Parse response
        question, hint = self._parse_response(response)

        return question, hint

    def _parse_response(self, response: str) -> tuple[str, str]:
        """Parse LLM response to extract question and hint."""
        # Try to extract JSON from response
        try:
            # Look for JSON in the response
            json_match = re.search(r'\{[^{}]*"question"[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("question", ""), data.get("hint", "")

            # Try parsing entire response as JSON
            data = json.loads(response.strip())
            return data.get("question", ""), data.get("hint", "")

        except json.JSONDecodeError:
            # Fallback: try to extract question and hint from text
            question = ""
            hint = ""

            # Look for question pattern
            q_match = re.search(r'"question"\s*:\s*"([^"]+)"', response)
            if q_match:
                question = q_match.group(1)

            # Look for hint pattern
            h_match = re.search(r'"hint"\s*:\s*"([^"]+)"', response)
            if h_match:
                hint = h_match.group(1)

            if question:
                return question, hint

            # Last resort: return error message
            return f"[VERBALIZATION FAILED: {response[:100]}]", ""

    async def verbalize_k(
        self,
        code: str,
        profile: dict,
        ground_truth: Any,
        output_schema: str = "",
        data_overview: str = "",
        dataset_description: str = "",
        k: int = 3,
    ) -> list[tuple[str, str]]:
        """
        Generate K diverse question variants for the same code.

        Args:
            code: The Python code that was executed
            profile: Dataset profile from DataProfiler
            ground_truth: The actual answer from executing the code
            output_schema: Description of the exact expected output format
            data_overview: Text summary of the dataset (from generate_data_overview)
            dataset_description: Human description of what the dataset contains
            k: Number of question variants to generate

        Returns:
            List of (question_text, hint) tuples
        """
        # Format ground truth for display
        if isinstance(ground_truth, dict):
            gt_str = json.dumps(ground_truth, indent=2)
        else:
            gt_str = str(ground_truth)

        # Build prompt
        prompt = VERBALIZATION_PROMPT_K.format(
            code=code.strip(),
            ground_truth=gt_str,
            output_schema=output_schema or "Not specified",
            data_overview=data_overview or "Not available",
            dataset_description=dataset_description or "No description provided",
            k=k,
        )

        # Call LLM
        response = await self.llm(prompt)

        # Parse K responses
        variants = self._parse_k_response(response, k)

        return variants

    def _parse_k_response(self, response: str, k: int) -> list[tuple[str, str]]:
        """Parse LLM response to extract K question/hint pairs."""
        variants = []

        try:
            # Try to find JSON array in response
            # Match array that may contain nested objects
            array_match = re.search(r'\[[\s\S]*\]', response)
            if array_match:
                data = json.loads(array_match.group())
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            q = item.get("question", "")
                            h = item.get("hint", "")
                            if q:
                                variants.append((q, h))
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract individual objects
        if not variants:
            # Look for individual {"question": ..., "hint": ...} objects
            obj_pattern = r'\{\s*"question"\s*:\s*"([^"]+)"\s*,\s*"hint"\s*:\s*"([^"]*)"\s*\}'
            matches = re.findall(obj_pattern, response)
            for q, h in matches:
                variants.append((q, h))

        # If we got fewer than requested, that's okay - return what we have
        if not variants:
            # Last resort: return single failed variant
            return [(f"[VERBALIZATION FAILED: {response[:100]}]", "")]

        return variants

    async def aclose(self):
        """Cleanup resources."""
        await self.llm.aclose()

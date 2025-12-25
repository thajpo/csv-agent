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

    async def aclose(self):
        """Cleanup resources."""
        await self.llm.aclose()

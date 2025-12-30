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


VERBALIZATION_PROMPT = '''You are writing questions for a data science training dataset.

## DATASET:
{dataset_description}

## DATA OVERVIEW:
{data_overview}

## THE ANALYSIS (what the question should lead to):
```python
{code}
```

## ANSWER:
{ground_truth}

## BANNED WORDS (DO NOT USE these terms in the question):
{banned_words}

## YOUR TASK:
Write a SHORT question (1-3 sentences) that expresses genuine curiosity and would lead someone to perform this analysis.

The agent should DISCOVER which method to use - don't give it away! Rephrase method-specific concepts:
- Instead of "bootstrap confidence interval" → "how certain can we be about this estimate?"
- Instead of "t-test" → "is there a significant difference between groups?"
- Instead of "regression" → "can we predict X from other features?"
- Instead of "Pearson correlation" → "how strongly related are these measurements?"

GOOD QUESTIONS (short, curious, don't describe the method):
- "Which numeric feature best predicts the target variable? Return as JSON, e.g.: {{"predictor": "<col>", "fit_score": 0.0}}"
- "I wonder if the most variable measurement can be explained by other features."
- "Are there any strong correlations between the clinical measurements?"
- "Does cholesterol differ significantly between outcome groups?"

BAD QUESTIONS (long, mechanical, describe the algorithm):
- "Identify the column with highest variance, then find the three most correlated columns, fit a regression, and report the R-squared..." ← TOO LONG, DESCRIBES STEPS
- "Compute the Pearson correlation between all numeric columns and return the pair with the highest absolute value..." ← DESCRIBES HOW TO DO IT

RULES:
1. Question must be 1-3 sentences MAX
2. Express curiosity, not instructions
3. Don't describe the method or steps - the agent must discover these
4. Don't mention specific column names
5. The hint can mention the approach
6. ALWAYS append output format with a placeholder example at the end. Use JSON for ALL outputs:
   - For dict outputs: "Return as JSON, e.g.: {{"column": "<name>", "mean": 0.0}}"
   - For scalar outputs: "Return as JSON, e.g.: {{"answer": 0.0}}"
   Use the REQUIRED OUTPUT KEYS below to construct the example.

OUTPUT (respond with ONLY this JSON):
{{
    "question": "Short curious question here. Return as JSON, e.g.: {{\"key\": \"<placeholder>\"}}",
    "hint": "Brief guidance on approach"
}}

REQUIRED OUTPUT KEYS: {output_schema}'''


class QuestionVerbalizer:
    """Convert code compositions into natural language questions using an LLM."""

    def __init__(self, model: str, sampling_args: dict):
        """
        Initialize the verbalizer.

        Args:
            model: Model identifier (see config.question_gen_model)
            sampling_args: Sampling configuration dict
        """
        self.llm = APILLM(model=model, sampling_args=sampling_args)

    async def verbalize(
        self,
        code: str,
        profile: dict,
        ground_truth: Any,
        output_schema: str = "",
        data_overview: str = "",
        dataset_description: str = "",
        banned_words: list[str] | None = None,
    ) -> tuple[str, str, str]:
        """
        Convert code into a natural language question.

        Args:
            code: The Python code that was executed
            profile: Dataset profile from DataProfiler
            ground_truth: The actual answer from executing the code
            output_schema: Description of the exact expected output format
            data_overview: Text summary of the dataset (from generate_data_overview)
            dataset_description: Human description of what the dataset contains
            banned_words: List of method terms to avoid in the question

        Returns:
            Tuple of (question_text, hint, raw_response)
        """
        # Format ground truth for display
        if isinstance(ground_truth, dict):
            gt_str = json.dumps(ground_truth, indent=2)
        else:
            gt_str = str(ground_truth)

        # Format banned words for prompt
        banned_str = ", ".join(banned_words) if banned_words else "None"

        # Build prompt with enriched context
        prompt = VERBALIZATION_PROMPT.format(
            code=code.strip(),
            ground_truth=gt_str,
            output_schema=output_schema or "Not specified",
            data_overview=data_overview or "Not available",
            dataset_description=dataset_description or "No description provided",
            banned_words=banned_str,
        )

        # Call LLM
        response = await self.llm(prompt)

        # Parse response
        question, hint = self._parse_response(response)

        return question, hint, response

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

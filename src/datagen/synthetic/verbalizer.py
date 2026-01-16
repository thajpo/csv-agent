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

## MANDATORY CONSTRAINT (VIOLATIONS CORRUPT TRAINING DATA):
The hint MUST match EXACTLY what the code does. DO NOT add exclusions or constraints not in the code.
- If code uses `df.select_dtypes('number')` → hint says "all numeric columns"
- NEVER say "excluding the target", "except the charge column", "without X" unless the code EXPLICITLY does this
- Look at the code: does it have `.drop()`, filtering, or exclusion logic? If NO, the hint cannot exclude anything.

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

BAD QUESTIONS:
- "Identify the column with highest variance, then find the three most correlated columns..." ← TOO LONG, DESCRIBES STEPS
- "Compute the Pearson correlation between all numeric columns and return..." ← DESCRIBES HOW TO DO IT

BAD HINTS (adding constraints not in code - THIS CORRUPTS TRAINING DATA):
- "Find the column with highest variance (excluding the target)..." ← WRONG if code doesn't exclude
- "Examine all numeric columns except the charge column..." ← WRONG if code uses all columns
- "Among the feature columns (not including the target)..." ← WRONG if code includes all

GOOD HINTS (matching what the code actually does):
- "Examine all numeric columns, find the one with highest variance..." ← CORRECT if code uses df.select_dtypes('number')
- "Compute variance for each numeric column and select the maximum..." ← CORRECT, no false exclusions

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
    "hint": "Brief guidance on approach (NO exclusions unless code excludes)"
}}

REQUIRED OUTPUT KEYS: {output_schema}'''


class QuestionVerbalizer:
    """Convert code compositions into natural language questions using an LLM."""

    # Phrases that indicate exclusion in hints
    EXCLUSION_PHRASES = [
        "excluding",
        "except the",
        "except for",
        "without the",
        "other than",
        "not including",
        "ignoring the",
    ]

    # Patterns in code that indicate intentional exclusion
    CODE_EXCLUSION_PATTERNS = [
        ".drop(",
        ".drop(columns",
        "!= ",
        "exclude",
        "remaining_cols",
        "feature_cols",  # Often used after excluding target
    ]

    def __init__(self, model: str, sampling_args: dict):
        """
        Initialize the verbalizer.

        Args:
            model: Model identifier (see config.question_gen_model)
            sampling_args: Sampling configuration dict
        """
        self.llm = APILLM(model=model, sampling_args=sampling_args)

    def _hint_has_spurious_exclusion(self, hint: str, code: str) -> bool:
        """
        Check if hint adds exclusion constraints not present in the code.

        Returns True if the hint is invalid (has exclusion but code doesn't).
        """
        hint_lower = hint.lower()
        code_lower = code.lower()

        # Check if hint contains exclusion language
        hint_has_exclusion = any(
            phrase in hint_lower for phrase in self.EXCLUSION_PHRASES
        )

        if not hint_has_exclusion:
            return False  # No exclusion in hint, it's fine

        # Hint has exclusion - check if code justifies it
        code_has_exclusion = any(
            pattern in code_lower for pattern in self.CODE_EXCLUSION_PATTERNS
        )

        # If hint has exclusion but code doesn't, it's spurious
        return not code_has_exclusion

    async def verbalize(
        self,
        code: str,
        profile: dict,
        ground_truth: Any,
        output_schema: str = "",
        data_overview: str = "",
        dataset_description: str = "",
        banned_words: list[str] | None = None,
        max_attempts: int = 3,
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
            max_attempts: Maximum attempts to generate a valid hint (default: 3)

        Returns:
            Tuple of (question_text, hint, raw_response)

        Raises:
            ValueError: If unable to generate a valid hint after max_attempts
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

        # Try multiple times to get a valid hint
        for attempt in range(max_attempts):
            response = await self.llm(prompt)
            question, hint = self._parse_response(response)

            # Validate hint doesn't add spurious exclusions
            is_spurious = self._hint_has_spurious_exclusion(hint, code)

            if not is_spurious:
                return question, hint, response

            # Log warning on retry
            print(
                f"[verbalizer] SPURIOUS EXCLUSION DETECTED (attempt {attempt + 1}/{max_attempts}): "
                f"{hint[:80]}..."
            )

        # All attempts failed - raise error
        raise ValueError(
            f"Could not generate valid hint after {max_attempts} attempts. "
            f"Last hint added exclusion not in code: {hint[:100]}"
        )

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

"""
Simple template-based question generator for tree growth dataset.

Generates questions by combining:
- Aggregations (mean, max, min, count)
- Groupings (by treatment, tree, branch)
- Comparisons (which group has highest/lowest)
- Filtering (subset of data)
"""

import pandas as pd
import random
from typing import List


# Template patterns for different question types
SIMPLE_AGG_TEMPLATES = [
    "What is the mean {target_col} for the {group_val} group?",
    "What is the maximum {target_col} in the {group_val} group?",
    "What is the minimum {target_col} for {group_val}?",
    "How many rows are in the {group_val} group?",
]

COMPARISON_TEMPLATES = [
    "Which {group_col} has the highest mean {target_col}?",
    "Which {group_col} has the lowest mean {target_col}?",
    "Which {group_col} has the most rows?",
    "Compare the mean {target_col} between {group_val1} and {group_val2}. Which is higher?",
]

MULTI_STEP_TEMPLATES = [
    "What is the difference in mean {target_col} between {group_val1} and {group_val2}?",
    "What percentage higher is the mean {target_col} for {group_val1} compared to {group_val2}?",
    "Find the {group_col} with the highest mean {target_col}, then calculate its standard deviation.",
]

FILTERING_TEMPLATES = [
    "For rows where {target_col} > {threshold}, what is the mean {target_col2}?",
    "How many {group_val} rows have {target_col} above {threshold}?",
    "What is the mean {target_col} for {group_val} where IN > {threshold}?",
]


def generate_simple_aggregation_questions(
    df: pd.DataFrame,
    n_questions: int = 5
) -> List[dict]:
    """Generate simple aggregation questions (mean, max, min, count)."""
    questions = []

    # Get available groups from TR column
    treatments = df['TR'].unique().tolist()
    numeric_cols = ['TL', 'IN']

    for _ in range(n_questions):
        template = random.choice(SIMPLE_AGG_TEMPLATES)
        treatment = random.choice(treatments)
        target_col = random.choice(numeric_cols)

        question = template.format(
            target_col=target_col,
            group_val=treatment
        )

        # Generate hint based on question type
        if "mean" in question:
            hint = f"Filter to {treatment}, then use .mean() on the {target_col} column."
        elif "maximum" in question:
            hint = f"Filter to {treatment}, then use .max() on the {target_col} column."
        elif "minimum" in question:
            hint = f"Filter to {treatment}, then use .min() on the {target_col} column."
        else:  # count
            hint = f"Filter to {treatment}, then count the rows with len()."

        questions.append({
            "question": question,
            "hint": hint,
            "difficulty": "EASY",
            "type": "simple_aggregation"
        })

    return questions


def generate_comparison_questions(
    df: pd.DataFrame,
    n_questions: int = 5
) -> List[dict]:
    """Generate comparison questions (which group is highest/lowest)."""
    questions = []

    treatments = df['TR'].unique().tolist()
    numeric_cols = ['TL', 'IN']

    for _ in range(n_questions):
        template = random.choice(COMPARISON_TEMPLATES)
        target_col = random.choice(numeric_cols)

        if "{group_val1}" in template:
            # Pick two different treatments
            group_val1, group_val2 = random.sample(treatments, 2)
            question = template.format(
                group_col="TR",
                target_col=target_col,
                group_val1=group_val1,
                group_val2=group_val2
            )
            hint = f"Group by TR, calculate mean {target_col} for {group_val1} and {group_val2}, then compare."
        else:
            question = template.format(
                group_col="TR",
                target_col=target_col
            )
            hint = f"Group by TR, calculate the aggregation on {target_col}, then find the max/min group."

        questions.append({
            "question": question,
            "hint": hint,
            "difficulty": "MEDIUM",
            "type": "comparison"
        })

    return questions


def generate_multi_step_questions(
    df: pd.DataFrame,
    n_questions: int = 3
) -> List[dict]:
    """Generate multi-step questions (difference, percentage, chained operations)."""
    questions = []

    treatments = df['TR'].unique().tolist()
    numeric_cols = ['TL', 'IN']

    for _ in range(n_questions):
        template = random.choice(MULTI_STEP_TEMPLATES)
        target_col = random.choice(numeric_cols)

        if "{group_val1}" in template:
            group_val1, group_val2 = random.sample(treatments, 2)
            question = template.format(
                group_col="TR",
                target_col=target_col,
                group_val1=group_val1,
                group_val2=group_val2
            )
            hint = f"Calculate mean {target_col} for both {group_val1} and {group_val2}, then compute the difference or percentage."
        else:
            question = template.format(
                group_col="TR",
                target_col=target_col
            )
            hint = f"First find the group with highest mean {target_col}, then calculate its std deviation."

        questions.append({
            "question": question,
            "hint": hint,
            "difficulty": "HARD",
            "type": "multi_step"
        })

    return questions


def generate_filtering_questions(
    df: pd.DataFrame,
    n_questions: int = 3
) -> List[dict]:
    """Generate questions with filtering conditions."""
    questions = []

    treatments = df['TR'].unique().tolist()
    numeric_cols = ['TL', 'IN']

    for _ in range(n_questions):
        template = random.choice(FILTERING_TEMPLATES)
        target_col = random.choice(numeric_cols)
        target_col2 = random.choice([c for c in numeric_cols if c != target_col])
        treatment = random.choice(treatments)

        # Calculate reasonable threshold (median of the column)
        threshold = int(df[target_col].median())

        question = template.format(
            target_col=target_col,
            target_col2=target_col2,
            group_val=treatment,
            threshold=threshold
        )

        hint = f"Filter rows where {target_col} > {threshold}, then calculate the aggregation."

        questions.append({
            "question": question,
            "hint": hint,
            "difficulty": "MEDIUM",
            "type": "filtering"
        })

    return questions


def generate_questions(
    csv_path: str = "data.csv",
    n_simple: int = 5,
    n_comparison: int = 5,
    n_multi_step: int = 3,
    n_filtering: int = 3,
    seed: int = 42
) -> List[dict]:
    """
    Generate a mix of questions for the dataset.

    Args:
        csv_path: Path to CSV file
        n_simple: Number of simple aggregation questions
        n_comparison: Number of comparison questions
        n_multi_step: Number of multi-step questions
        n_filtering: Number of filtering questions
        seed: Random seed for reproducibility

    Returns:
        List of question dicts with fields: question, hint, difficulty, type
    """
    random.seed(seed)
    df = pd.read_csv(csv_path)

    questions = []
    questions.extend(generate_simple_aggregation_questions(df, n_simple))
    questions.extend(generate_comparison_questions(df, n_comparison))
    questions.extend(generate_multi_step_questions(df, n_multi_step))
    questions.extend(generate_filtering_questions(df, n_filtering))

    # Shuffle to mix difficulty levels
    random.shuffle(questions)

    return questions


if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Generate questions for CSV dataset")
    parser.add_argument("--csv", default="data.csv", help="Path to CSV file")
    parser.add_argument("--output", "-o", default="questions.json", help="Output JSON file")
    parser.add_argument("--n-simple", type=int, default=5, help="Number of simple questions")
    parser.add_argument("--n-comparison", type=int, default=5, help="Number of comparison questions")
    parser.add_argument("--n-multi-step", type=int, default=3, help="Number of multi-step questions")
    parser.add_argument("--n-filtering", type=int, default=3, help="Number of filtering questions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    questions = generate_questions(
        csv_path=args.csv,
        n_simple=args.n_simple,
        n_comparison=args.n_comparison,
        n_multi_step=args.n_multi_step,
        n_filtering=args.n_filtering,
        seed=args.seed
    )

    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(questions, f, indent=2)

    print(f"Generated {len(questions)} questions")
    print(f"Saved to {args.output}")
    print("\nSample questions:")
    for i, q in enumerate(questions[:3], 1):
        print(f"\n{i}. [{q['difficulty']}] {q['question']}")
        print(f"   Hint: {q['hint']}")

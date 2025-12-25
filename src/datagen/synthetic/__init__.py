"""Synthetic question generation via code composition templates."""

from src.datagen.synthetic.generator import (
    CompositionalQuestionGenerator,
    generate_questions,
)
from src.datagen.synthetic.templates import (
    CompositionTemplate,
    get_applicable_templates,
)

__all__ = [
    "CompositionalQuestionGenerator",
    "generate_questions",
    "CompositionTemplate",
    "get_applicable_templates",
]

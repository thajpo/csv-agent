"""
Conversation management for CSV agent.

Simplified conversation tracking with automatic context pruning to stay
within token limits. Uses simple Message objects instead of complex Turn tracking.
"""

from pydantic import BaseModel, ConfigDict, Field
from dataclasses import dataclass
from typing import Any, TypedDict


# ============= TypedDict for Messages =============

class MessageDict(TypedDict):
    """OpenAI message format."""
    role: str  # "system", "user", "assistant"
    content: str


# ============= Code Execution Result (still needed by some code) =============

class CodeCellResult(BaseModel):
    """Result of executing one code cell."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    code: str
    success: bool
    stdout: str
    stderr: str
    submitted_answer: Any | None = None


# ============= Simple Message Representation =============

@dataclass
class Message:
    """A single message in the conversation (OpenAI format)."""
    role: str  # "system", "user", "assistant"
    content: str


# ============= Simplified Conversation History =============

class ConversationHistory(BaseModel):
    """
    Manages conversation messages with context pruning.
    """
    system_prompt: str
    messages: list[MessageDict] = Field(default_factory=list)  # OpenAI format with type hints
    archived_count: int = 0           # Number of messages archived

    # Limits
    max_messages: int = 10       # Max active messages (2 msgs/turn)
    max_context_tokens: int = 80_000

    def add_assistant_response(self, response: str) -> None:
        """Add assistant message."""
        self.messages.append({"role": "assistant", "content": response})
        self._maybe_prune()

    def add_user_feedback(self, feedback: str) -> None:
        """Add user message."""
        self.messages.append({"role": "user", "content": feedback})
        self._maybe_prune()

    def _maybe_prune(self) -> None:
        """Prune oldest messages if over limits."""
        # Token-based pruning
        total_tokens = self._estimate_tokens()
        while total_tokens > self.max_context_tokens and len(self.messages) > 2:
            self.messages.pop(0)
            self.archived_count += 1
            total_tokens = self._estimate_tokens()

        # Message count pruning
        while len(self.messages) > self.max_messages:
            self.messages.pop(0)
            self.archived_count += 1

    def _estimate_tokens(self) -> int:
        """Rough token estimate (4 chars = 1 token)."""
        total = len(self.system_prompt) // 4
        for msg in self.messages:
            total += len(msg["content"]) // 4
        return total

    def to_openai_messages(self) -> list[dict]:
        """Convert to OpenAI API format."""
        return [
            {"role": "system", "content": self.system_prompt},
            *self.messages
        ]

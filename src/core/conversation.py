"""
Conversation management for CSV agent.

Simplified conversation tracking with automatic context pruning to stay
within token limits. Uses simple Message objects instead of complex Turn tracking.
"""

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
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

    Uses incremental token counting for O(1) pruning instead of O(n²).
    """
    system_prompt: str
    messages: list[MessageDict] = Field(default_factory=list)
    archived_count: int = 0
    _cached_message_tokens: int = PrivateAttr(default=0)  # Incremental token count

    # Limits - these are set by Environment.init_state() from config
    # Default values here are fallbacks only (config.max_active_turns * 2)
    max_messages: int = 10       # Max active messages (2 msgs/turn)
    max_context_tokens: int = 80_000

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _tokens_for_content(self, content: str) -> int:
        """Estimate tokens for content (4 chars = 1 token)."""
        return len(content) // 4

    def add_assistant_response(self, response: str) -> None:
        """Add assistant message."""
        self.messages.append({"role": "assistant", "content": response})
        self._cached_message_tokens += self._tokens_for_content(response)
        self._maybe_prune()

    def add_user_feedback(self, feedback: str) -> None:
        """Add user message."""
        self.messages.append({"role": "user", "content": feedback})
        self._cached_message_tokens += self._tokens_for_content(feedback)
        self._maybe_prune()

    def _total_tokens(self) -> int:
        """Total tokens including system prompt."""
        return self._tokens_for_content(self.system_prompt) + self._cached_message_tokens

    def _maybe_prune(self) -> None:
        """Prune oldest messages if over limits. O(n) worst case, O(1) amortized."""
        # Token-based pruning
        while self._total_tokens() > self.max_context_tokens and len(self.messages) > 2:
            removed = self.messages.pop(0)
            self._cached_message_tokens -= self._tokens_for_content(removed["content"])
            self.archived_count += 1

        # Message count pruning
        while len(self.messages) > self.max_messages:
            removed = self.messages.pop(0)
            self._cached_message_tokens -= self._tokens_for_content(removed["content"])
            self.archived_count += 1

    def _estimate_tokens(self) -> int:
        """Rough token estimate (4 chars = 1 token). For backwards compatibility."""
        return self._total_tokens()

    def to_openai_messages(self) -> list[dict]:
        """Convert to OpenAI API format."""
        return [
            {"role": "system", "content": self.system_prompt},
            *self.messages
        ]

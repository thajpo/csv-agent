"""
Environment class for CSV agent.

This is a pure RL-style environment that executes episodes (rollouts)
for CSV exploration and question generation. It uses Python's logging
module for output, keeping the environment logic separate from presentation.

Refactored to use a sandboxed Python environment for code execution.
"""

import ast
import json
import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

from src.core.model import APILLM
from src.utils.parsing import (
    get_turn_validation_feedback,
    parse_execution_result,
    extract_python_cells,
)
from src.core.prompts import (
    generate_data_overview,
    build_system_prompt,
    CONTINUE_MSG,
    FINAL_MSG,
)
from src.core.config import DataConfig, ModelConfig, ExecutionConfig, TaskConfig
from src.core.conversation import CodeCellResult, ConversationHistory
from src.envs.csv_env import LocalCSVAnalysisEnv as CSVAnalysisEnv

# Max output chars before truncation (~12.5K tokens at 4 chars/token)
MAX_OUTPUT_CHARS = 50_000


def validate_hooks_grounded(
    hooks: list[dict], code_cells: list[str]
) -> tuple[list[dict], list[dict]]:
    """
    Validate that each hook's code_line is grounded in the executed code.

    A hook is "grounded" if its code_line appears as a substring in any executed code cell.
    This prevents the model from hallucinating code_lines that weren't actually run.

    Args:
        hooks: List of hook dicts with code_line field
        code_cells: List of executed code strings

    Returns:
        Tuple of (grounded_hooks, ungrounded_hooks)
    """
    # Concatenate all code cells for searching
    all_code = "\n".join(code_cells)

    grounded = []
    ungrounded = []

    for hook in hooks:
        code_line = hook.get("code_line", "")
        if not code_line:
            ungrounded.append(hook)
            continue

        # Normalize whitespace for matching (strip leading/trailing, collapse internal)
        normalized_code_line = " ".join(code_line.split())
        normalized_all_code = " ".join(all_code.split())

        if normalized_code_line in normalized_all_code:
            grounded.append(hook)
        else:
            ungrounded.append(hook)

    return grounded, ungrounded


HOOK_REPROMPT_MSG = """
⚠️ YOUR SOLUTION WAS REJECTED - HOOKS ARE MISSING OR INVALID

Your submission must include hook() calls that document each computational step.
Each hook's code_line MUST be the EXACT code that was executed.

REQUIRED PATTERN:
```python
# Step 1: Filter data
filtered = df[df['col'] == 'value']
hook(filtered, "filtered = df[df['col'] == 'value']", name='filtered')

# Step 2: Compute result
result = filtered['amount'].mean()
hook(result, "result = filtered['amount'].mean()", name='result', depends_on=['filtered'])

submit(result)
```

Please re-do your solution with proper hook() calls after EVERY computational step.
The code_line argument must EXACTLY match the code you wrote.
"""


def parse_submitted_answer(output: str) -> str | None:
    """
    Extract submitted answer from execution output.

    Looks for "✓ Submitted: {answer}" pattern in stdout.
    The answer is expected to be JSON-serialized.

    Returns:
        The submitted answer value, or None if no submission found
    """
    match = re.search(r"✓ Submitted: (.+)", output)
    if not match:
        return None

    answer_str = match.group(1).strip()

    # Try JSON first (expected format)
    try:
        return json.loads(answer_str)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try Python literal (legacy format)
    try:
        return ast.literal_eval(answer_str)
    except (ValueError, SyntaxError):
        pass

    # Fallback to raw string - log the unexpected format
    logger.warning(
        f"Answer not parseable as JSON/literal, returning raw string: {answer_str[:100]}"
    )
    return answer_str


# Keywords that suggest a statistical/hypothesis answer needing structured format
_STATISTICAL_KEYWORDS = {
    "yes",
    "no",
    "significant",
    "not significant",
    "reject",
    "fail to reject",
    "accept",
}


def _needs_structured_format(answer) -> bool:
    """
    Check if a submitted answer looks like it should be structured but isn't.

    Returns True if the answer is a string containing statistical keywords,
    suggesting the model should have submitted a dict with answer/p_value.
    """
    if not isinstance(answer, str):
        return False

    answer_lower = answer.lower().strip()

    # If it's already a simple value (number, short label), accept it
    if len(answer_lower) < 20 and not any(
        kw in answer_lower for kw in _STATISTICAL_KEYWORDS
    ):
        return False

    # If it contains statistical keywords, it should be structured
    return any(kw in answer_lower for kw in _STATISTICAL_KEYWORDS)


FORMAT_REPROMPT_MSG = """
⚠️ Your answer appears to be a statistical conclusion but was submitted as a plain string.

Please re-submit using the structured format:
```python
submit({"answer": "Yes", "p_value": 0.0012})
```

Replace "Yes" with your conclusion and the p-value with your computed value.
"""


class Environment:
    """
    RL-style environment for CSV exploration.

    This class handles the execution of multi-turn episodes where
    an LLM explores a CSV dataset using tools. It's designed to be
    pure RL logic with no presentation dependencies (uses stdlib logging).

    Uses a sandboxed Python environment for code execution.
    """

    def __init__(
        self,
        data: DataConfig,
        model: ModelConfig,
        execution: ExecutionConfig,
        task: TaskConfig,
        env: CSVAnalysisEnv | None = None,
        state: dict | None = None,
        reuse_env: bool = False,
        llm=None,
    ):
        # Store configs
        self.data = data
        self.model_config = model
        self.execution = execution
        self.task = task

        self.csv_path = data.csv_path
        if llm is not None:
            self.model = llm
        else:
            self.model = APILLM(
                model=model.model_name, sampling_args=model.sampling_args_dict()
            )
        self.env = env
        self.state = state
        self.reuse_env = reuse_env

        self.df = None
        self.submitted_answer = None

    @classmethod
    async def create(
        cls,
        data: DataConfig,
        model: ModelConfig,
        execution: ExecutionConfig,
        task: TaskConfig,
        env: CSVAnalysisEnv | None = None,
        state: dict | None = None,
        reuse_env: bool = False,
        llm=None,
    ):
        instance = cls(data, model, execution, task, env, state, reuse_env, llm)

        # Create env and state if not provided
        if instance.env is None:
            instance.env = CSVAnalysisEnv(csv_path=instance.csv_path)
            instance.state = {}
            instance.state = await instance.env.setup_state(instance.state)
        elif instance.state is None:
            # Env provided but no state - set up state
            instance.state = {}
            instance.state = await instance.env.setup_state(instance.state)

        return instance

    @classmethod
    async def from_params(
        cls,
        csv_path: str,
        model: str,
        *,
        question: str | None = None,
        hint: str | None = None,
        n_steps: int | None = None,
        difficulty: str | None = None,
        mode: str = "teacher-tutor",
        dataset_description: str = "",
        data_overview: str = "",
        max_turns: int = 10,
        sampling_args: dict,
        env: CSVAnalysisEnv | None = None,
        state: dict | None = None,
        reuse_env: bool = False,
        llm=None,
    ):
        """
        Factory with primitive args - handles config construction internally.

        This is the preferred way to create an Environment. Callers pass primitives,
        and this method builds the config objects internally.

        Args:
            csv_path: Path to CSV file
            model: Model identifier (see config.teacher_model)
            question: Question text (optional)
            hint: Hint for the question (optional)
            n_steps: Expected number of solution steps/hooks
            difficulty: Question difficulty (EASY, MEDIUM, HARD, VERY_HARD)
            mode: Execution mode (teacher-tutor, teacher-consistency, student)
            dataset_description: Description of the dataset
            data_overview: Generated data overview string
            max_turns: Maximum conversation turns
            sampling_args: Dict of temperature, max_tokens, top_p (optional)
            env: Optional pre-created LocalCSVAnalysisEnv (for pooling)
            state: Optional pre-created state dict (for pooling)
            reuse_env: If True, reset env after rollout instead of destroying

        Returns:
            Initialized Environment ready for rollout
        """
        from src.core.types import Question

        # Build question object if provided
        question_obj = (
            Question(
                question_text=question,
                hint=hint,
                n_steps=n_steps,
                difficulty=difficulty,
            )
            if question
            else None
        )

        # Build configs from primitives
        data_config = DataConfig(
            csv_path=csv_path,
            dataset_description=dataset_description,
            data_overview=data_overview,
        )

        model_config = ModelConfig(model_name=model, **sampling_args)

        execution_config = ExecutionConfig(
            max_turns=max_turns,
        )

        task_config = TaskConfig(
            mode=mode,
            question=question_obj,
        )

        return await cls.create(
            data=data_config,
            model=model_config,
            execution=execution_config,
            task=task_config,
            env=env,
            state=state,
            reuse_env=reuse_env,
            llm=llm,
        )

    def _load_csv(self):
        """Load CSV file if not already loaded."""
        if self.df is None:
            try:
                self.df = pd.read_csv(self.csv_path)
            except UnicodeDecodeError:
                self.df = pd.read_csv(self.csv_path, encoding="latin-1")

    def init_state(self):
        self._load_csv()
        data_overview = generate_data_overview(self.csv_path)
        sys_prompt = build_system_prompt(
            mode=self.task.mode,
            dataset_description=self.data.dataset_description,
            data_overview=self.data.data_overview,
            question=self.task.question,
        )

        # Create conversation history with context management
        conversation = ConversationHistory(
            system_prompt=sys_prompt,
            max_messages=self.execution.max_active_turns * 2,  # 2 messages per turn
            max_context_tokens=self.execution.max_context_tokens,
        )

        # Initialize episode state as instance variables
        self.conversation = conversation
        self.current_turn = 0
        self.is_completed = False
        self.data_overview = data_overview
        self.submitted_answer = None  # Reset for new episode
        self.submission_metadata = {}  # Metadata (key_lines, etc.)
        self.code_cells = []  # Track all executed code cells
        self.execution_results_per_turn = []  # Track execution results per turn
        self.format_reprompt_count = 0  # Track format re-prompts (force-accept after 3)
        self.hook_reprompt_count = 0  # Track hook re-prompts (force-accept after 3)

    def extract_python_cells(self, response: str) -> list[str]:
        """Extract ```python...``` code blocks from response."""
        return extract_python_cells(response)

    async def execute_code_cell(self, code: str) -> CodeCellResult:
        """
        Execute code in CSVAnalysisEnv sandbox and return execution result.
        """
        # Execute via verifiers env.python()
        output = await self.env.python(
            code=code,
            sandbox_id=self.state["sandbox_id"],
            python_state=self.state["python_state"],
        )

        # Truncate massive outputs to prevent context overflow
        # Preserve the ✓ Submitted: line intact (it contains the answer JSON)
        if len(output) > MAX_OUTPUT_CHARS:
            submit_marker = "✓ Submitted:"
            submit_idx = output.find(submit_marker)

            if submit_idx != -1:
                # Find end of submission line
                submit_end = output.find("\n", submit_idx)
                if submit_end == -1:
                    submit_end = len(output)
                submit_line = output[submit_idx:submit_end]

                # If submit_line itself is too large, progressively truncate
                max_submit_len = MAX_OUTPUT_CHARS - 5000  # Leave room for context
                if len(submit_line) > max_submit_len:
                    json_start = submit_line.find("{")
                    if json_start != -1:
                        try:
                            submit_json = json.loads(submit_line[json_start:])

                            # Step 1: Truncate hooks to empty list
                            if "hooks" in submit_json:
                                submit_json["hooks"] = []
                            truncated_json = json.dumps(submit_json, default=str)
                            submit_line = submit_line[:json_start] + truncated_json
                            logger.warning(
                                f"Truncated hooks in submission (was too large for context)"
                            )

                            # Step 2: If still too large, the answer itself is huge
                            if len(submit_line) > max_submit_len:
                                answer = submit_json.get("__csv_agent_answer__")
                                answer_str = json.dumps(answer, default=str)

                                # Replace answer with a marker dict (NOT string)
                                # This preserves protocol but ensures triangulation fails
                                submit_json["__csv_agent_answer__"] = {
                                    "__answer_truncated__": True,
                                    "reason": "Answer too large to preserve",
                                }
                                truncated_json = json.dumps(submit_json, default=str)
                                submit_line = submit_line[:json_start] + truncated_json
                                logger.warning(
                                    f"Answer value too large ({len(answer_str):,} chars), replaced with truncation marker"
                                )
                        except json.JSONDecodeError:
                            # Can't parse - just truncate the line
                            submit_line = submit_line[:max_submit_len] + "...[TRUNCATED]"
                            logger.warning(
                                f"Submission line too large and not parseable, truncating"
                            )

                # Keep start + submission line
                # CRITICAL: Don't include any part of the original submission in output[:keep_start]
                # Otherwise the parser will find the truncated original instead of our clean version
                keep_start = max(0, MAX_OUTPUT_CHARS - len(submit_line) - 100)
                keep_start = min(keep_start, submit_idx)  # Never include original submission
                truncated_chars = len(output) - keep_start - len(submit_line)
                logger.warning(
                    f"Truncating output: {len(output):,} chars -> ~{keep_start + len(submit_line):,} chars "
                    f"(removed {truncated_chars:,} chars, preserved submission)"
                )
                output = (
                    output[:keep_start] +
                    f"\n\n... [TRUNCATED {truncated_chars:,} chars] ...\n\n" +
                    submit_line
                )
            else:
                # No submission found, use middle-out truncation
                truncated_chars = len(output) - MAX_OUTPUT_CHARS
                keep_each = MAX_OUTPUT_CHARS // 2
                logger.warning(
                    f"Truncating output: {len(output):,} chars -> {MAX_OUTPUT_CHARS:,} chars "
                    f"(removed {truncated_chars:,} chars from middle)"
                )
                output = (
                    output[:keep_each] +
                    f"\n\n... [TRUNCATED {truncated_chars:,} chars] ...\n\n" +
                    output[-keep_each:]
                )

        # Parse the string output into success/stdout/stderr
        result = parse_execution_result(output)
        result.code = code

        # Check for submitted answer in output
        submitted = parse_submitted_answer(output)
        if submitted is not None:
            # Enforce strict protocol: answer MUST be wrapped
            if isinstance(submitted, dict) and "__csv_agent_answer__" in submitted:
                self.submitted_answer = submitted["__csv_agent_answer__"]
                self.submission_metadata = submitted
            else:
                # Protocol violation: answer not wrapped
                logger.error(
                    f"Protocol violation: Answer submitted without wrapper. "
                    f"Expected {{'__csv_agent_answer__': value}}, got {type(submitted).__name__}. "
                    f"Agent must use submit() function."
                )
                raise ValueError(
                    "Answer must be submitted via submit() function. "
                    f"Received unwrapped {type(submitted).__name__} instead of protocol dict."
                )

            result.submitted_answer = self.submitted_answer

        return result

    async def handle_max_turns_reached(self) -> None:
        """Handle reaching max turns: prompt for final output and get response."""

        self.conversation.add_user_feedback(FINAL_MSG)

        response = await self.get_model_response()

        self.conversation.add_assistant_response(response)
        self.is_completed = True

    async def get_model_response(self) -> str:
        """Call model and log the interaction."""
        messages = self.conversation.to_openai_messages()

        try:
            response = await self.model(messages)
        except Exception as e:
            # Enrich error with context for debugging
            context = self._get_error_context(messages)
            raise RuntimeError(f"{e}\n\n[Context] {context}") from e

        return response

    def _get_error_context(self, messages: list[dict]) -> str:
        """Build context string for error messages."""
        from pathlib import Path

        csv_name = Path(self.csv_path).stem if self.csv_path else "unknown"
        q = self.task.question
        question_id = (q.id or q.generate_id()) if q else "unknown"
        question_text = (q.question_text[:50] + "...") if q else "unknown"
        worker_id = self.state.get("sandbox_id", "unknown") if self.state else "unknown"
        turn = getattr(self, "current_turn", "?")

        # Estimate tokens (4 chars ~ 1 token)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        est_tokens = total_chars // 4

        return (
            f"csv={csv_name}, question_id={question_id}, "
            f"turn={turn}, worker={worker_id}, est_tokens={est_tokens:,}, "
            f"question='{question_text}'"
        )

    def response_is_valid(self, response: str, code_cells: list[str]) -> bool:
        """Response should have reasoning text and one code cell."""
        error_msg = get_turn_validation_feedback(response, code_cells)
        if error_msg:
            self.conversation.add_assistant_response(response)
            error_feedback = (
                error_msg + "\n\nPlease try again following the correct format."
            )
            self.conversation.add_user_feedback(error_feedback)
            return False
        return True

    # ============= Process Turn Helpers =============

    async def _execute_cells(
        self, code_cells: list[str]
    ) -> tuple[list[CodeCellResult], any]:
        """Execute code cells and return results with any submitted answer."""
        results = []
        submitted = None

        for cell_code in code_cells:
            result = await self.execute_code_cell(cell_code)
            results.append(result)
            self.code_cells.append(cell_code)

            if result.submitted_answer is not None:
                submitted = result.submitted_answer
                break  # Stop on submit()

        return results, submitted

    def _build_execution_feedback(
        self, code_cells: list[str], results: list[CodeCellResult]
    ) -> str:
        """Build feedback string from execution results."""
        if not code_cells:
            return "No code blocks found. Write Python code in ```python blocks."

        parts = []
        for i, result in enumerate(results, 1):
            if result.success:
                parts.append(f"✓ Cell {i} executed successfully")
                if result.stdout.strip():
                    parts.append(f"Output:\n{result.stdout}")
            else:
                parts.append(f"✗ Cell {i} failed")
                parts.append(f"Error:\n{result.stderr}")

        return "\n\n".join(parts) + CONTINUE_MSG

    def _validate_format(self, answer: any) -> tuple[bool, str | None]:
        """Check if answer format is valid. Returns (valid, error_feedback)."""
        if not _needs_structured_format(answer):
            return True, None

        self.format_reprompt_count += 1
        if self.format_reprompt_count < 3:
            return False, FORMAT_REPROMPT_MSG

        # Force-accept after 3 retries
        return True, None

    def _validate_hooks(self) -> tuple[bool, str | None]:
        """Check if hooks are grounded and sufficient. Returns (valid, error_feedback).

        Force-accepts after 3 failed validation attempts to prevent infinite loops.
        """
        if self.hook_reprompt_count >= 3:
            return True, None  # Already at max retries, force-accept

        hooks = self.submission_metadata.get("hooks", [])
        # Ensure hooks is a list (could be string if truncated in edge cases)
        if not isinstance(hooks, list):
            hooks = []
        grounded, ungrounded = validate_hooks_grounded(hooks, self.code_cells)

        # Get expected hook count from question
        expected = 2  # Default minimum
        if self.task and self.task.question and self.task.question.n_steps:
            expected = self.task.question.n_steps

        has_ungrounded = len(ungrounded) > 0
        insufficient = len(grounded) < expected

        if not has_ungrounded and not insufficient:
            return True, None

        self.hook_reprompt_count += 1
        if self.hook_reprompt_count >= 3:
            return True, None  # Just hit max retries, force-accept

        # Build error feedback
        parts = [HOOK_REPROMPT_MSG]
        if has_ungrounded:
            parts.append(
                f"\n❌ Found {len(ungrounded)} ungrounded hook(s) - "
                f"code_line does not match any executed code."
            )
        if insufficient:
            parts.append(
                f"\n❌ Expected ~{expected} hooks but found only {len(grounded)} valid hook(s)."
            )

        # Clear state for retry
        self.code_cells = []

        return False, "".join(parts)

    # ============= Main Process Turn =============

    async def process_turn(self, response: str) -> None:
        """Process a single turn: execute code, validate, update conversation."""
        code_cells = self.extract_python_cells(response)

        # Execute
        results, submitted = (
            await self._execute_cells(code_cells) if code_cells else ([], None)
        )
        feedback = self._build_execution_feedback(code_cells, results)

        # Validate submission
        done = submitted is not None
        if done:
            valid, error = self._validate_format(submitted)
            if not valid:
                feedback = error
                done = False
                self.submitted_answer = None
                self.submission_metadata = {}  # Symmetric cleanup

        if done:
            valid, error = self._validate_hooks()
            if not valid:
                feedback = error
                done = False
                self.submitted_answer = None
                self.submission_metadata = {}

        # Update state
        self.execution_results_per_turn.append(results)
        self.conversation.add_assistant_response(response)
        self.conversation.add_user_feedback(feedback)

        if done:
            self.is_completed = True

    async def rollout(self):
        """Execute a multi-turn rollout episode.

        Returns:
            self: The Environment instance with completed conversation
        """
        self.init_state()

        try:
            while not self.is_completed:
                # Check if we've reached max turns BEFORE processing
                if self.current_turn >= self.execution.max_turns:
                    await self.handle_max_turns_reached()
                    break

                # Get model response
                response = await self.get_model_response()

                # Extract code cells and validate
                code_cells = self.extract_python_cells(response)
                if not self.response_is_valid(response, code_cells):
                    # Don't increment turn counter - let model retry
                    continue

                # Process this turn (adds to conversation, executes code cells)
                await self.process_turn(response)

                # Increment turn counter
                self.current_turn += 1

        finally:
            # Cleanup or reset sandbox
            if self.state and "sandbox_id" in self.state:
                if self.reuse_env:
                    # Reset for reuse instead of destroying
                    await self.env.reset(
                        self.state["sandbox_id"], self.state.get("python_state")
                    )
                else:
                    await self.env.destroy_sandbox(self.state["sandbox_id"])

        return self

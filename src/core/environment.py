"""
Environment class for CSV agent.

This is a pure RL-style environment that executes episodes (rollouts)
for CSV exploration and question generation. It uses Python's logging
module for output, keeping the environment logic separate from presentation.

Refactored to use a sandboxed Python environment for code execution.
"""

import json
import logging
from copy import deepcopy
from pathlib import Path

import pandas as pd
from csv_spec import (
    TrajectoryPrefix,
    TurnDict,
    parse_hook_record,
    validate_hook_event_line,
)

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
)
from src.datagen.shared.submission import parse_submission, validate_submission_position
from src.core.config import DataConfig, ModelConfig, ExecutionConfig, TaskConfig
from src.core.conversation import CodeCellResult, ConversationHistory
from src.envs.csv_env import LocalCSVAnalysisEnv as CSVAnalysisEnv

logger = logging.getLogger(__name__)

# Max output chars before truncation (~12.5K tokens at 4 chars/token)
MAX_OUTPUT_CHARS = 50_000


class PrefixReplayError(RuntimeError):
    """Raised when a recorded prefix cannot be reproduced faithfully."""


def _hook_record_identity(hook: dict) -> str:
    fields = (
        "variable_name",
        "code_line",
        "value",
        "value_hash",
        "depends_on",
        "description",
        "event_line",
    )
    return json.dumps(
        {field: hook.get(field) for field in fields},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _parse_hook_records(
    output: str, *, code: str, trusted_records: list[dict] | None = None
) -> list[dict]:
    """Capture hooks while distinguishing runtime events from stdout records."""
    stdout_hooks: list[tuple[dict, str]] = []
    for line in output.splitlines():
        if "📍 Hook:" not in line:
            continue
        json_start = line.find("{")
        if json_start == -1:
            continue
        try:
            payload = json.loads(line[json_start:])
        except json.JSONDecodeError:
            continue
        hook = parse_hook_record(payload)
        if hook is None:
            continue
        stdout_hooks.append((hook, _hook_record_identity(payload)))

    trusted_hooks: list[dict] = []
    matched_stdout_indexes: set[int] = set()
    for payload in trusted_records or []:
        hook = parse_hook_record(payload, trusted_event_provenance=True)
        if hook is None:
            continue
        event_line = validate_hook_event_line(code, hook.get("event_line"))
        hook["event_line"] = event_line
        if event_line is None:
            hook["event_provenance_reason"] = "missing_or_ambiguous_event_provenance"
        identity = _hook_record_identity(payload)
        for index, (_stdout_hook, stdout_identity) in enumerate(stdout_hooks):
            if index not in matched_stdout_indexes and stdout_identity == identity:
                matched_stdout_indexes.add(index)
                break
        trusted_hooks.append(hook)

    diagnostics = [
        hook
        for index, (hook, _identity) in enumerate(stdout_hooks)
        if index not in matched_stdout_indexes
    ]
    return trusted_hooks + diagnostics


def validate_hooks_grounded(
    hooks: list[dict], code_cells: list[str]
) -> tuple[list[dict], list[dict]]:
    """
    Validate that each hook's code_line is grounded in the executed code.

    A hook is "grounded" if each line of its code_line exactly matches a
    logical line from the executed code (after whitespace normalization).
    This prevents the model from hallucinating code_lines or getting false
    positives from substring matches (e.g., "x = 1" should NOT match "x = 10").

    Args:
        hooks: List of hook dicts with code_line field
        code_cells: List of executed code strings

    Returns:
        Tuple of (grounded_hooks, ungrounded_hooks)
    """
    # Build set of normalized code lines for exact matching
    # Each line is normalized: strip leading/trailing whitespace, collapse internal whitespace
    normalized_code_lines = set()
    for cell in code_cells:
        for line in cell.split("\n"):
            normalized = " ".join(line.split())
            if normalized:  # Skip empty lines
                normalized_code_lines.add(normalized)

    grounded = []
    ungrounded = []

    for hook in hooks:
        code_line = hook.get("code_line", "")
        if not code_line:
            hook["_ungrounded_reason"] = "missing code_line"
            ungrounded.append(hook)
            continue

        normalized_hook_lines = [
            normalized
            for hook_line in code_line.split("\n")
            if (normalized := " ".join(hook_line.split()))
        ]
        if not normalized_hook_lines:
            hook["_ungrounded_reason"] = "missing code_line"
            ungrounded.append(hook)
            continue

        all_lines_match = all(
            normalized_hook_line in normalized_code_lines
            for normalized_hook_line in normalized_hook_lines
        )

        if all_lines_match:
            grounded.append(hook)
        else:
            hook["_ungrounded_reason"] = "code_line not found in executed code"
            ungrounded.append(hook)

    return grounded, ungrounded


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

    Uses a sandboxed Python environment for code execution. Recorded
    nonterminal prefixes can be replayed in a fresh sandbox before continuing;
    replay raises ``PrefixReplayError`` if execution differs from the record.
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
        session_id: str | None = None,
        system_prompt_suffix: str | None = None,
    ):
        # Store configs
        self.data = data
        self.model_config = model
        self.execution = execution
        self.task = task
        self.session_id = session_id
        self.system_prompt_suffix = system_prompt_suffix

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
        session_id: str | None = None,
        system_prompt_suffix: str | None = None,
    ):
        instance = cls(
            data,
            model,
            execution,
            task,
            env,
            state,
            reuse_env,
            llm,
            session_id,
            system_prompt_suffix,
        )

        # Create env and state if not provided
        if instance.env is None:
            instance.env = CSVAnalysisEnv(
                csv_path=instance.csv_path, session_id=session_id
            )
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
        session_id: str | None = None,
        system_prompt_suffix: str | None = None,
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
            session_id: Session ID for container isolation (for parallel execution)
            system_prompt_suffix: Optional experiment instruction appended verbatim

        Returns:
            Initialized Environment ready for rollout
        """
        from csv_spec import Question

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
            session_id=session_id,
            system_prompt_suffix=system_prompt_suffix,
        )

    def _load_csv(self):
        """Load CSV file if not already loaded."""
        if self.df is None:
            try:
                self.df = pd.read_csv(
                    self.csv_path,
                    na_values=["?", "NA", "N/A", "na", "n/a"],
                    keep_default_na=True,
                )
            except UnicodeDecodeError:
                self.df = pd.read_csv(
                    self.csv_path,
                    encoding="latin-1",
                    na_values=["?", "NA", "N/A", "na", "n/a"],
                    keep_default_na=True,
                )

    def init_state(self):
        self._load_csv()
        data_overview = generate_data_overview(self.csv_path)
        sys_prompt = build_system_prompt(
            mode=self.task.mode,
            dataset_description=self.data.dataset_description,
            data_overview=self.data.data_overview,
            question=self.task.question,
        )
        if self.system_prompt_suffix:
            sys_prompt = f"{sys_prompt}\n\n{self.system_prompt_suffix.strip()}"

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
        self.execution_turns = []
        self.format_reprompt_count = 0  # Track format re-prompts (force-accept after 3)

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
        trusted_records = self.state["python_state"].pop("hooks", None)
        hooks = _parse_hook_records(
            output,
            code=code,
            trusted_records=trusted_records,
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
                                "Truncated hooks in submission (was too large for context)"
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
                            submit_line = (
                                submit_line[:max_submit_len] + "...[TRUNCATED]"
                            )
                            logger.warning(
                                "Submission line too large and not parseable, truncating"
                            )

                # Keep start + submission line
                # CRITICAL: Don't include any part of the original submission in output[:keep_start]
                # Otherwise the parser will find the truncated original instead of our clean version
                keep_start = max(0, MAX_OUTPUT_CHARS - len(submit_line) - 100)
                keep_start = min(
                    keep_start, submit_idx
                )  # Never include original submission
                truncated_chars = len(output) - keep_start - len(submit_line)
                logger.warning(
                    f"Truncating output: {len(output):,} chars -> ~{keep_start + len(submit_line):,} chars "
                    f"(removed {truncated_chars:,} chars, preserved submission)"
                )
                output = (
                    output[:keep_start]
                    + f"\n\n... [TRUNCATED {truncated_chars:,} chars] ...\n\n"
                    + submit_line
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
                    output[:keep_each]
                    + f"\n\n... [TRUNCATED {truncated_chars:,} chars] ...\n\n"
                    + output[-keep_each:]
                )

        # Parse the string output into success/stdout/stderr
        result = parse_execution_result(output)
        result.code = code
        result.hooks = hooks

        # Check for submitted answer in output
        submission, success = parse_submission(output)
        if success and submission is not None:
            # Enforce strict protocol: answer MUST be wrapped
            if isinstance(submission, dict) and "__csv_agent_answer__" in submission:
                self.submitted_answer = submission["__csv_agent_answer__"]
                self.submission_metadata = submission
            else:
                # Protocol violation: answer not wrapped
                logger.error(
                    f"Protocol violation: Answer submitted without wrapper. "
                    f"Expected {{'__csv_agent_answer__': value}}, got {type(submission).__name__}. "
                    f"Agent must use submit() function."
                )
                raise ValueError(
                    "Answer must be submitted via submit() function. "
                    f"Received unwrapped {type(submission).__name__} instead of protocol dict."
                )

            result.submitted_answer = self.submitted_answer

        return result

    async def get_model_response(self) -> str:
        """Call model and log the interaction."""
        messages = self.conversation.to_openai_messages()
        if len(messages) == 1:
            messages.append({"role": "user", "content": "Begin the analysis."})

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
        if not error_msg and code_cells:
            try:
                validate_submission_position(code_cells[0])
            except ValueError as error:
                error_msg = str(error)
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

        # Update state
        self.conversation.add_assistant_response(response)
        self.conversation.add_user_feedback(feedback)
        self.execution_turns.append(
            {
                "response": response,
                "code_cells": list(code_cells),
                "execution_results": results,
                "completed": done,
                "conversation_messages": deepcopy(self.conversation.messages),
                "consumed_turns": self.current_turn + 1,
            }
        )

        if done:
            self.is_completed = True

    @staticmethod
    def _execution_for_comparison(result: CodeCellResult) -> dict:
        return {
            "success": result.success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "hooks": result.hooks,
            "submitted_answer": result.submitted_answer,
        }

    @staticmethod
    def _replay_values_equal(left: object, right: object) -> bool:
        """Compare JSON-like execution values with stable NaN handling."""
        return json.dumps(
            left,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ) == json.dumps(
            right,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    async def replay_turns(
        self,
        turns: list[TurnDict],
        turn_responses: list[str],
        conversation_messages: list[dict[str, str]],
        consumed_turns: int,
    ) -> None:
        """Restore a public turn boundary after exact execution replay.

        Each recorded response must contain one valid code cell. Success,
        stdout, stderr, hooks, and submitted answer are compared exactly; any
        divergence or terminal replay raises ``PrefixReplayError``. After the
        checks pass, the exact recorded conversation feedback is restored.
        """
        if not hasattr(self, "conversation"):
            raise RuntimeError("init_state() must be called before replay_turns()")
        if len(turns) != len(turn_responses):
            raise PrefixReplayError("recorded turn responses do not align with turns")

        for expected_index, (turn, response) in enumerate(
            zip(turns, turn_responses, strict=True)
        ):
            if turn.get("turn_index") != expected_index:
                raise PrefixReplayError(
                    f"turn {expected_index} is not contiguous in recorded prefix"
                )

            code_cells = self.extract_python_cells(response)
            validation_error = get_turn_validation_feedback(response, code_cells)
            if validation_error:
                raise PrefixReplayError(
                    f"turn {expected_index} cannot be replayed: {validation_error}"
                )

            await self.process_turn(response)
            execution_results = self.execution_turns[-1]["execution_results"]
            if len(execution_results) != 1:
                raise PrefixReplayError(
                    f"turn {expected_index} replayed {len(execution_results)} cells; expected 1"
                )

            actual = self._execution_for_comparison(execution_results[0])
            expected = turn.get("execution")
            if not self._replay_values_equal(actual, expected):
                differing_fields = [
                    field
                    for field in actual
                    if not self._replay_values_equal(
                        actual.get(field), (expected or {}).get(field)
                    )
                ]
                raise PrefixReplayError(
                    f"turn {expected_index} replay diverged in: "
                    f"{', '.join(differing_fields) or 'execution record'}"
                )

            if self.is_completed:
                raise PrefixReplayError(
                    f"turn {expected_index} submitted an answer and is not a prefix"
                )
            self.current_turn += 1

        self.current_turn = consumed_turns
        self.conversation.messages = deepcopy(conversation_messages)
        self.conversation._cached_message_tokens = sum(
            self.conversation._tokens_for_content(message["content"])
            for message in conversation_messages
        )

    async def _continue_rollout(self) -> None:
        """Continue from the currently initialized conversation and sandbox."""
        while not self.is_completed:
            if self.current_turn >= self.execution.max_turns:
                self.is_completed = True
                break

            response = await self.get_model_response()
            code_cells = self.extract_python_cells(response)
            if not self.response_is_valid(response, code_cells):
                self.current_turn += 1
                continue

            await self.process_turn(response)
            self.current_turn += 1

    async def _cleanup_sandbox(self) -> None:
        if self.state and "sandbox_id" in self.state:
            if self.reuse_env:
                await self.env.reset(
                    self.state["sandbox_id"], self.state.get("python_state")
                )
            else:
                await self.env.destroy_sandbox(self.state["sandbox_id"])

    async def rollout_from_prefix(self, prefix: TrajectoryPrefix) -> "Environment":
        """Replay a matching prefix in a fresh sandbox, then finish its rollout.

        The configured question, CSV path, and maximum turn budget must match
        the prefix. Sandbox cleanup follows the same rules as ``rollout``.
        """
        configured_question = (
            self.task.question.question_text if self.task.question else ""
        )
        if configured_question != prefix.question_text:
            raise ValueError("environment question does not match prefix")
        if self.execution.max_turns != prefix.max_turns:
            raise ValueError("environment max_turns does not match prefix")
        if Path(self.csv_path).resolve() != Path(prefix.csv_source).resolve():
            raise ValueError("environment CSV does not match prefix")

        self.init_state()
        try:
            self.conversation.system_prompt = prefix.system_prompt
            await self.replay_turns(
                prefix.turns,
                prefix.turn_responses,
                prefix.conversation_messages,
                prefix.consumed_turns,
            )
            await self._continue_rollout()
        finally:
            await self._cleanup_sandbox()
        return self

    async def rollout(self):
        """Execute a multi-turn rollout episode.

        Returns:
            self: The Environment instance with completed conversation
        """
        self.init_state()

        try:
            await self._continue_rollout()
        finally:
            await self._cleanup_sandbox()

        return self

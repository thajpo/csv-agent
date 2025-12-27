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
from src.utils.interaction import (
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
    logger.warning(f"Answer not parseable as JSON/literal, returning raw string: {answer_str[:100]}")
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
                self.df = pd.read_csv(self.csv_path, encoding='latin-1')

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
                import logging

                logging.error(
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
        response = await self.model(messages)

        return response

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

    async def process_turn(self, response: str) -> None:
        """Process a single turn: extract code cells, execute, build feedback, check completion."""
        code_cells = self.extract_python_cells(response)

        # Execute all cells
        execution_results = []
        submitted_answer = None

        if code_cells:
            for cell_code in code_cells:
                result = await self.execute_code_cell(cell_code)

                execution_results.append(result)
                self.code_cells.append(cell_code)

                # Log execution

                # Check for submission
                if result.submitted_answer is not None:
                    submitted_answer = result.submitted_answer
                    break  # Stop on submit()

        # 4. Build feedback from execution results
        if code_cells:
            feedback_parts = []
            for i, result in enumerate(execution_results, 1):
                if result.success:
                    feedback_parts.append(f"✓ Cell {i} executed successfully")
                    if result.stdout.strip():
                        feedback_parts.append(f"Output:\n{result.stdout}")
                else:
                    feedback_parts.append(f"✗ Cell {i} failed")
                    feedback_parts.append(f"Error:\n{result.stderr}")

            feedback = "\n\n".join(feedback_parts)
            feedback += CONTINUE_MSG
        else:
            feedback = "No code blocks found. Write Python code in ```python blocks."

        # 5. Check for completion (submit() was called)
        done_signal = submitted_answer is not None

        # 5a. Format validation: if answer needs structured format, re-prompt (up to 3 times)
        if done_signal and _needs_structured_format(submitted_answer):
            self.format_reprompt_count += 1
            if self.format_reprompt_count < 3:
                # Re-prompt for correct format
                feedback = FORMAT_REPROMPT_MSG
                done_signal = False
                self.submitted_answer = None  # Clear so they can re-submit
            # else: force-accept after 3 retries

        # 5b. Hook validation: hooks must be grounded and sufficient
        if done_signal and self.hook_reprompt_count < 3:
            hooks = self.submission_metadata.get("hooks", [])

            # Check grounding: code_line must be substring of executed code
            grounded_hooks, ungrounded_hooks = validate_hooks_grounded(
                hooks, self.code_cells
            )

            # Get expected hook count from question if available
            expected_hooks = 2  # Minimum default
            if self.task and self.task.question and self.task.question.n_steps:
                expected_hooks = self.task.question.n_steps

            # Fail conditions: ungrounded hooks OR insufficient hook count
            has_ungrounded = len(ungrounded_hooks) > 0
            insufficient_hooks = len(grounded_hooks) < expected_hooks

            if has_ungrounded or insufficient_hooks:
                self.hook_reprompt_count += 1
                if self.hook_reprompt_count < 3:
                    # Build specific feedback
                    feedback_parts = [HOOK_REPROMPT_MSG]
                    if has_ungrounded:
                        feedback_parts.append(
                            f"\n❌ Found {len(ungrounded_hooks)} ungrounded hook(s) - "
                            f"code_line does not match any executed code."
                        )
                    if insufficient_hooks:
                        feedback_parts.append(
                            f"\n❌ Expected ~{expected_hooks} hooks but found only {len(grounded_hooks)} valid hook(s)."
                        )
                    feedback = "".join(feedback_parts)
                    done_signal = False
                    self.submitted_answer = None
                    self.submission_metadata = {}
                    # Clear code_cells so model must re-run everything with hooks
                    self.code_cells = []
                # else: force-accept after 3 retries

        # 6. Store execution results for this turn
        self.execution_results_per_turn.append(execution_results)

        # 7. Add to conversation (response + feedback)
        self.conversation.add_assistant_response(response)
        self.conversation.add_user_feedback(feedback)

        # 8. Check completion
        if done_signal:
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

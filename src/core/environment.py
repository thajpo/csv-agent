"""
Environment class for CSV agent.

This is a pure RL-style environment that executes episodes (rollouts)
for CSV exploration and question generation. It uses Python's logging
module for output, keeping the environment logic separate from presentation.

Refactored to use a sandboxed Python environment for code execution.
"""


import re

import pandas as pd

from src.core.model import APILLM
from src.core.model import APILLM
from src.utils.interaction import get_turn_validation_feedback, parse_execution_result, extract_python_cells
from src.core.prompts import generate_data_overview, build_system_prompt, CONTINUE_MSG, FINAL_MSG
from src.core.config import DataConfig, ModelConfig, ExecutionConfig, TaskConfig
from src.core.conversation import CodeCellResult, ConversationHistory
from src.envs.csv_env import LocalCSVAnalysisEnv as CSVAnalysisEnv


def truncate_output(text: str, max_length: int = 500) -> str:
    """Truncate execution output to max_length chars."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"\n... (truncated {len(text) - max_length} chars)"


def parse_submitted_answer(output: str) -> str | None:
    """
    Extract submitted answer from execution output.

    Looks for "✓ Submitted: {answer}" pattern in stdout.
    The answer is expected to be JSON-serialized.

    Returns:
        The submitted answer value, or None if no submission found
    """
    # Pattern: "✓ Submitted: <value>"
    match = re.search(r"✓ Submitted: (.+)", output)
    if match:
        answer_str = match.group(1).strip()
        # Try to parse as JSON first (new format)
        try:
            import json
            return json.loads(answer_str)
        except (json.JSONDecodeError, ValueError):
            # Fall back to ast.literal_eval for backward compatibility
            try:
                import ast
                return ast.literal_eval(answer_str)
            except (ValueError, SyntaxError):
                # Return as string if can't parse
                return answer_str
    return None


# Keywords that suggest a statistical/hypothesis answer needing structured format
_STATISTICAL_KEYWORDS = {'yes', 'no', 'significant', 'not significant', 'reject', 'fail to reject', 'accept'}

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
    if len(answer_lower) < 20 and not any(kw in answer_lower for kw in _STATISTICAL_KEYWORDS):
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
    ):
        """
        Initialize Environment with focused configs.

        Args:
            data: Dataset paths and metadata
            model: LLM model and sampling parameters
            execution: Execution limits (turns, tokens, context)
            task: Task definition (mode, question)
            env: Optional CSVAnalysisEnv (created if None)

        """
        # Store configs
        self.data = data
        self.model_config = model
        self.execution = execution
        self.task = task

        # Derived values
        self.csv_path = data.csv_path
        self.model = APILLM(
            model=model.model_name,
            sampling_args=model.sampling_args_dict()
        )
        self.env = env  # Will be created in create() if None
        self.state = state  # Verifiers state dict
        self.reuse_env = reuse_env  # If True, reset instead of destroy

        self.df = None  # Will be loaded on first rollout
        
        # Track submitted answer across executions
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
    ):
        """Async factory to create Environment with initialized CSVAnalysisEnv."""
        instance = cls(data, model, execution, task, env, state, reuse_env)
        
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
        mode: str = "teacher-tutor",
        dataset_description: str = "",
        data_overview: str = "",
        max_turns: int = 10,
        sampling_args: dict | None = None,
        env: CSVAnalysisEnv | None = None,
        state: dict | None = None,
        reuse_env: bool = False,
    ):
        """
        Factory with primitive args - handles config construction internally.
        
        This is the preferred way to create an Environment. Callers pass primitives,
        and this method builds the config objects internally.
        
        Args:
            csv_path: Path to CSV file
            model: Model identifier (e.g., 'openai/gpt-4o')
            question: Question text (optional)
            hint: Hint for the question (optional)
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
        question_obj = Question(question_text=question, hint=hint) if question else None
        
        # Build configs from primitives
        data_config = DataConfig(
            csv_path=csv_path,
            dataset_description=dataset_description,
            data_overview=data_overview,
        )
        
        model_config = ModelConfig(
            model_name=model,
            **(sampling_args or {})
        )
        
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
        )

    def _load_csv(self):
        """Load CSV file if not already loaded."""
        if self.df is None:
            self.df = pd.read_csv(self.csv_path)


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
        self.data_overview = data_overview
        self.submitted_answer = None  # Reset for new episode
        self.submission_metadata = {}  # Metadata (key_lines, etc.)
        self.code_cells = []  # Track all executed code cells
        self.execution_results_per_turn = []  # Track execution results per turn
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
            sandbox_state=self.state["sandbox_state"],
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
            error_feedback = error_msg + "\n\nPlease try again following the correct format."
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

                # Truncate outputs to save memory
                result.stdout = truncate_output(result.stdout, max_length=500)
                result.stderr = truncate_output(result.stderr, max_length=500)

                execution_results.append(result)
                self.code_cells.append(cell_code)  # Track executed code

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
                try:
                    if self.reuse_env:
                        # Reset for reuse instead of destroying
                        await self.env.reset(
                            self.state["sandbox_id"],
                            self.state.get("python_state")
                        )
                    else:
                        await self.env.destroy_sandbox(self.state["sandbox_id"])
                except Exception as e:
                    pass  # Silently ignore cleanup failures

        return self

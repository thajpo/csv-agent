"""
Environment class for CSV agent.

This is a pure RL-style environment that executes episodes (rollouts)
for CSV exploration and question generation. It uses Python's logging
module for output, keeping the environment logic separate from presentation.

Refactored to use verifiers CSVAnalysisEnv instead of JupyterKernel.
"""


import re

import pandas as pd

from src.core.model import APILLM
from src.core.validation import get_turn_validation_feedback
from src.core.prompts import generate_data_overview, build_system_prompt, CONTINUE_MSG, FINAL_MSG
from src.core.config import DataConfig, ModelConfig, ExecutionConfig, TaskConfig
from src.core.conversation import CodeCellResult, ConversationHistory
from src.envs.csv_env import CSVAnalysisEnv


def parse_execution_result(output: str) -> tuple[bool, str, str]:
    """
    Parse verifiers PythonEnv output string into (success, stdout, stderr).
    
    The verifiers format returns:
    - stdout lines
    - "stderr:\n..." if there's stderr
    - Traceback if there's an error
    - "Out[N]: ..." for results
    - "(no output)" if empty
    
    Returns:
        (success, stdout, error_message)
    """
    if not output or output == "(no output)":
        return True, "", ""
    
    # Check for error indicators (Python traceback)
    error_indicators = [
        "Traceback (most recent call last):",
        "Error:",
        "Exception:",
    ]
    
    is_error = any(indicator in output for indicator in error_indicators)
    
    if is_error:
        # Extract the error portion
        return False, "", output
    
    # Check for stderr section
    if "stderr:\n" in output:
        parts = output.split("stderr:\n", 1)
        stdout = parts[0].strip()
        stderr = parts[1].strip() if len(parts) > 1 else ""
        # stderr doesn't always mean failure
        return True, stdout, stderr
    
    # Normal output
    return True, output, ""


def parse_submitted_answer(output: str) -> str | None:
    """
    Extract submitted answer from execution output.
    
    Looks for "✓ Submitted: {answer}" pattern in stdout.
    
    Returns:
        The submitted answer value, or None if no submission found
    """
    # Pattern: "✓ Submitted: <value>"
    match = re.search(r"✓ Submitted: (.+)", output)
    if match:
        answer_str = match.group(1).strip()
        # Try to eval it back to Python object
        try:
            import ast
            return ast.literal_eval(answer_str)
        except (ValueError, SyntaxError):
            # Return as string if can't parse
            return answer_str
    return None


class Environment:
    """
    RL-style environment for CSV exploration.

    This class handles the execution of multi-turn episodes where
    an LLM explores a CSV dataset using tools. It's designed to be
    pure RL logic with no presentation dependencies (uses stdlib logging).
    
    Uses verifiers CSVAnalysisEnv for sandboxed code execution.
    """

    def __init__(
        self,
        data: DataConfig,
        model: ModelConfig,
        execution: ExecutionConfig,
        task: TaskConfig,
        env: CSVAnalysisEnv | None = None,

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
            sampling_args=model.sampling_args()
        )
        self.env = env  # Will be created in create() if None
        self.state = None  # Verifiers state dict

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

    ):
        """Async factory to create Environment with initialized CSVAnalysisEnv."""
        instance = cls(data, model, execution, task, env)
        
        # Create env and state if not provided
        if instance.env is None:
            instance.env = CSVAnalysisEnv(csv_path=instance.csv_path)
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
        self.submitted_answer = None  # Reset for new episode
        self.code_cells = []  # Track all executed code cells

    def extract_python_cells(self, response: str) -> list[str]:
        """Extract ```python...``` code blocks from response."""
        pattern = r"```python\n(.*?)```"
        return re.findall(pattern, response, re.DOTALL)

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
        success, stdout, stderr = parse_execution_result(output)
        
        # Check for submitted answer in output
        submitted = parse_submitted_answer(output)
        if submitted is not None:
            self.submitted_answer = submitted

        return CodeCellResult(
            code=code,
            success=success,
            stdout=stdout if success else output,  # Full output for display
            stderr=stderr if not success else "",
            submitted_answer=self.submitted_answer,
        )

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

        # 6. Add to conversation (response + feedback)
        self.conversation.add_assistant_response(response)
        self.conversation.add_user_feedback(feedback)

        # 7. Check completion
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
            # Cleanup sandbox
            if self.state and "sandbox_id" in self.state:
                try:
                    await self.env.destroy_sandbox(self.state["sandbox_id"])
                except Exception as e:
                    pass  # Silently ignore cleanup failures

        return self

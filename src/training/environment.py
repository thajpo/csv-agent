"""
Environment class for CSV agent.

This is a pure RL-style environment that executes episodes (rollouts)
for CSV exploration and question generation. It uses Python's logging
module for output, keeping the environment logic separate from presentation.
"""

import logging
import re
from datetime import datetime

import pandas as pd

from src.core.model import APILLM
from src.utils.rich_logger import LogContext
from src.authoring.prompts import generate_data_overview
from src.core.prompts import RolloutConfig
from src.core.types import EnvironmentConfig, StateConfig
from src.core.conversation import Turn, CodeCellResult, ConversationManager
from src.core.kernel import JupyterKernel


def truncate_output_lines(text: str, max_line_length: int = 200) -> str:
    """Truncate each line of output to max_line_length characters."""
    lines = text.split('\n')
    truncated_lines = []
    for line in lines:
        if len(line) > max_line_length:
            truncated_lines.append(line[:max_line_length] + "... [truncated]")
        else:
            truncated_lines.append(line)
    return '\n'.join(truncated_lines)


def validate_turn_structure(response: str, code_cells: list[str]) -> tuple[bool, str]:
    """
    Validate that turn follows required pattern: reasoning text + exactly one code block.

    Args:
        response: Full model response
        code_cells: Extracted code cells from response

    Returns:
        (is_valid, error_message) - error_message is empty string if valid
    """
    # Check for exactly one code block
    if len(code_cells) == 0:
        return False, "❌ No code block found. You must write exactly ONE ```python code block."

    if len(code_cells) > 1:
        return False, f"❌ Found {len(code_cells)} code blocks. Write exactly ONE ```python code block per turn."

    # Check for reasoning text before code
    # Extract text before first code block
    code_block_pattern = r'```python\n.*?```'
    parts = re.split(code_block_pattern, response, maxsplit=1, flags=re.DOTALL)

    reasoning_text = parts[0].strip() if parts else ""

    # Require at least some minimal reasoning (not just whitespace)
    if not reasoning_text or len(reasoning_text) < 10:
        return False, "❌ Write your reasoning first: explain what you'll do and why (1-3 sentences), then write the code block."

    return True, ""


class Environment:
    """
    RL-style environment for CSV exploration.

    This class handles the execution of multi-turn episodes where
    an LLM explores a CSV dataset using tools. It's designed to be
    pure RL logic with no presentation dependencies (uses stdlib logging).
    """

    def __init__(
        self,
        csv_path: str = "data.csv",
        config: EnvironmentConfig | None = None,
        sampling_args: dict | None = None,
        rollout_config: RolloutConfig | None = None,
        kernel: JupyterKernel | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize Environment.

        Args:
            csv_path: Path to CSV file
            config: Environment configuration
            sampling_args: Sampling arguments for the model
            rollout_config: Rollout configuration (system prompt, messages)
            kernel: Jupyter kernel for code execution (None = create new one)
            logger: Optional logger for output (None = silent execution)
        """
        self.csv_path = csv_path
        self.config = config or EnvironmentConfig()
        self.rollout_config = rollout_config
        self.model = APILLM(model=self.config.model, sampling_args=sampling_args or {})
        self.kernel = kernel or JupyterKernel(timeout=120.0, csv_path=csv_path)
        self.logger = logger
        self.df = None  # Will be loaded on first rollout

    def _load_csv(self):
        """Load CSV file if not already loaded."""
        if self.df is None:
            self.df = pd.read_csv(self.csv_path)
            if self.logger:
                self.logger.info(
                    "csv_loaded",
                    extra={
                        "csv_path": self.csv_path,
                        "rows": len(self.df),
                        "cols": len(self.df.columns),
                    },
                )

    def init_state(self):
        if self.logger:
            self.logger.info("episode_start", extra={"csv_path": self.csv_path})

        self._load_csv()
        data_overview = generate_data_overview(self.csv_path)
        sys_prompt = self.build_system_prompt()

        # Create conversation manager with context management
        conversation_manager = ConversationManager(
            system_prompt=sys_prompt,
            max_active_turns=self.config.max_active_turns,
            max_context_tokens=self.config.max_context_tokens
        )

        return StateConfig(
            input=data_overview,
            conversation_manager=conversation_manager,
            n_turns=self.config.max_turns,
            is_completed=False,
            current_turn=0,
        )

    def build_system_prompt(self) -> str:
        """Build system prompt from rollout config."""
        return self.rollout_config.system_prompt

    def extract_python_cells(self, response: str) -> list[str]:
        """Extract ```python...``` code blocks from response."""
        pattern = r'```python\n(.*?)```'
        return re.findall(pattern, response, re.DOTALL)

    def execute_code_cell(self, code: str) -> CodeCellResult:
        """
        Execute code in kernel and return execution result.

        Returns CodeCellResult with success, stdout, stderr, and submitted_answer.
        """
        # Execute in kernel
        result = self.kernel.execute(code)

        # Check for submitted answer
        submitted_answer = self.kernel.get_final_answer()

        return CodeCellResult(
            code=code,
            success=result.success,
            stdout=result.stdout,
            stderr=result.stderr if not result.success else "",
            submitted_answer=submitted_answer
        )

    def handle_max_turns_reached(self, state: StateConfig) -> None:
        """Handle reaching max turns: prompt for final output and get response."""
        self.logger.info("max_turns_reached", extra={"max_turns": state.n_turns})

        # Create a special turn for the final message prompt
        final_prompt_turn = Turn(
            turn_number=state.current_turn,
            timestamp=datetime.now(),
            model_response="",  # No model response yet
            done_signal=False,
            feedback_message=self.rollout_config.final_msg,
            reasoning=None
        )
        state.conversation_manager.add_turn(final_prompt_turn)

        # Get final response
        with LogContext(self.logger, "model_thinking"):
            messages = state.conversation_manager.to_openai_messages()
            response = self.model(messages)

        self.logger.info("model_response", extra={"response": response})

        # Add final response as another turn
        final_response_turn = Turn(
            turn_number=state.current_turn + 1,
            timestamp=datetime.now(),
            model_response=response,
            done_signal=True,
            feedback_message="",
            reasoning=None
        )
        state.conversation_manager.add_turn(final_response_turn)

        state.is_completed = True

    def get_model_response(self, state: StateConfig) -> str:
        """Call model and log the interaction."""
        with LogContext(self.logger, "model_thinking"):
            messages = state.conversation_manager.to_openai_messages()
            response = self.model(messages)

        self.logger.info("model_response", extra={"response": response})
        return response

    def process_turn(self, state: StateConfig, response: str) -> None:
        """
        Process a single turn: extract code cells, execute, build feedback, check completion.

        Modifies state in-place.
        """
        # 1. Extract Python cells from response
        code_cells = self.extract_python_cells(response)

        # 2. Validate turn structure
        is_valid, error_msg = validate_turn_structure(response, code_cells)
        if not is_valid:
            # Give model clear feedback to try again
            error_turn = Turn(
                turn_number=state.current_turn,
                timestamp=datetime.now(),
                model_response=response,
                code_cells=[],
                execution_results=[],
                done_signal=False,
                feedback_message=error_msg + "\n\nPlease try again following the correct format.",
                reasoning=None,
            )
            state.conversation_manager.add_turn(error_turn)
            # Don't increment turn counter - let model retry
            return

        # 3. Execute all cells
        execution_results = []
        submitted_answer = None

        if code_cells:
            for cell_code in code_cells:
                result = self.execute_code_cell(cell_code)
                execution_results.append(result)

                # Log execution
                if self.logger:
                    self.logger.info("code_executed", extra={
                        "success": result.success,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    })

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
                        # Truncate stdout to 200 chars per line
                        truncated_stdout = truncate_output_lines(result.stdout)
                        feedback_parts.append(f"Output:\n{truncated_stdout}")
                else:
                    feedback_parts.append(f"✗ Cell {i} failed")
                    # Truncate stderr to 200 chars per line
                    truncated_stderr = truncate_output_lines(result.stderr)
                    feedback_parts.append(f"Error:\n{truncated_stderr}")

            feedback = "\n\n".join(feedback_parts)
            feedback += self.rollout_config.continue_msg
        else:
            feedback = "No code blocks found. Write Python code in ```python blocks."
            if self.logger:
                self.logger.info("no_code_blocks")

        # 5. Check for completion (submit() was called)
        done_signal = (submitted_answer is not None)

        # 6. Create Turn object
        turn = Turn(
            turn_number=state.current_turn,
            timestamp=datetime.now(),
            model_response=response,
            code_cells=code_cells,
            execution_results=execution_results,
            done_signal=done_signal,
            feedback_message=feedback,
            reasoning=None,
        )

        # 7. Add turn to conversation manager (auto-purges if needed)
        state.conversation_manager.add_turn(turn)

        # 8. Check completion
        if done_signal:
            state.is_completed = True
            if self.logger:
                self.logger.info("episode_complete", extra={"results": submitted_answer})

    def rollout(self) -> StateConfig:
        """Execute a multi-turn rollout episode."""
        state = self.init_state()

        while not state.is_completed:
            self.logger.info("turn_start", extra={
                "turn": state.current_turn,
                "max_turns": state.n_turns
            })

            # Check if we've reached max turns BEFORE processing
            if state.current_turn >= state.n_turns:
                self.handle_max_turns_reached(state)
                break

            # Get model response
            response = self.get_model_response(state)

            # Process this turn (adds to conversation, executes code cells)
            self.process_turn(state, response)

            # Increment turn counter
            state.current_turn += 1

        return state

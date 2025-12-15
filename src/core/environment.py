"""
Environment class for CSV agent.

This is a pure RL-style environment that executes episodes (rollouts)
for CSV exploration and question generation. It uses Python's logging
module for output, keeping the environment logic separate from presentation.
"""

import logging
import re

import pandas as pd

from src.core.model import APILLM
from src.utils.logger import create_logger
from src.utils.validation import get_turn_validation_feedback
from src.core.prompts import generate_data_overview, build_system_prompt, CONTINUE_MSG, FINAL_MSG
from src.core.config import DataConfig, ModelConfig, ExecutionConfig, TaskConfig
from src.core.conversation import CodeCellResult, ConversationHistory
from src.core.kernel import JupyterKernel


class Environment:
    """
    RL-style environment for CSV exploration.

    This class handles the execution of multi-turn episodes where
    an LLM explores a CSV dataset using tools. It's designed to be
    pure RL logic with no presentation dependencies (uses stdlib logging).
    """

    def __init__(
        self,
        data: DataConfig,
        model: ModelConfig,
        execution: ExecutionConfig,
        task: TaskConfig,
        kernel: JupyterKernel | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize Environment with focused configs.

        Args:
            data: Dataset paths and metadata
            model: LLM model and sampling parameters
            execution: Execution limits (turns, tokens, context)
            task: Task definition (mode, question)
            kernel: Optional JupyterKernel (created if None)
            logger: Optional logger (created if None)
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
        self.kernel = kernel  # Will be created in create() if None
        self.logger = logger or create_logger(silent=True)
        self.df = None  # Will be loaded on first rollout

    @classmethod
    async def create(
        cls,
        data: DataConfig,
        model: ModelConfig,
        execution: ExecutionConfig,
        task: TaskConfig,
        kernel: JupyterKernel | None = None,
        logger: logging.Logger | None = None,
    ):
        """Async factory to create Environment with initialized kernel."""
        env = cls(data, model, execution, task, kernel, logger)
        
        # Create kernel if not provided
        if env.kernel is None:
            env.kernel = await JupyterKernel.create(
                timeout=120.0,
                csv_path=env.csv_path,
                workdir="."
            )
        
        return env

    def _load_csv(self):
        """Load CSV file if not already loaded."""
        if self.df is None:
            self.df = pd.read_csv(self.csv_path)
            self.logger.info(
                "csv_loaded",
                extra={
                    "csv_path": self.csv_path,
                    "rows": len(self.df),
                    "cols": len(self.df.columns),
                },
            )

    def init_state(self):
        self.logger.info("episode_start", extra={"csv_path": self.csv_path})

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

    def extract_python_cells(self, response: str) -> list[str]:
        """Extract ```python...``` code blocks from response."""
        pattern = r"```python\n(.*?)```"
        return re.findall(pattern, response, re.DOTALL)

    async def execute_code_cell(self, code: str) -> CodeCellResult:
        """
        Execute code in kernel and return execution result.
        """
        # Execute in kernel
        result = await self.kernel.execute(code)

        # Check for submitted answer
        submitted_answer = await self.kernel.get_final_answer()

        return CodeCellResult(
            code=code,
            success=result.success,
            stdout=result.stdout,
            stderr=result.stderr if not result.success else "",
            submitted_answer=submitted_answer,
        )

    async def handle_max_turns_reached(self) -> None:
        """Handle reaching max turns: prompt for final output and get response."""
        self.logger.info("max_turns_reached", extra={"max_turns": self.execution.max_turns})
        self.conversation.add_user_feedback(FINAL_MSG)

        response = await self.get_model_response()

        self.conversation.add_assistant_response(response)
        self.is_completed = True

    async def get_model_response(self) -> str:
        """Call model and log the interaction."""
        messages = self.conversation.to_openai_messages()
        response = await self.model(messages)

        self.logger.info("model_response", extra={"response": response})
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

                # Log execution
                self.logger.info(
                    "code_executed",
                    extra={
                        "success": result.success,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    },
                )

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
            self.logger.info("no_code_blocks")

        # 5. Check for completion (submit() was called)
        done_signal = submitted_answer is not None

        # 6. Add to conversation (response + feedback)
        self.conversation.add_assistant_response(response)
        self.conversation.add_user_feedback(feedback)

        # 7. Check completion
        if done_signal:
            self.is_completed = True
            self.logger.info("episode_complete", extra={"results": submitted_answer})

    async def rollout(self):
        """Execute a multi-turn rollout episode.

        Returns:
            self: The Environment instance with completed conversation
        """
        self.init_state()

        while not self.is_completed:
            self.logger.info(
                "turn_start",
                extra={"turn": self.current_turn, "max_turns": self.execution.max_turns},
            )

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

        return self

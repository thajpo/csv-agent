"""
Environment class for CSV agent.

This is a pure RL-style environment that executes episodes (rollouts)
for CSV exploration and question generation. It uses Python's logging
module for output, keeping the environment logic separate from presentation.
"""

import json
import logging
import re
from datetime import datetime

import pandas as pd

from src.model import APILLM
from src.rich_logger import LogContext
from src.prompts import generate_data_overview, RolloutConfig
from src.types import EnvironmentConfig, StateConfig
from src.conversation import Turn, ConversationManager


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
        config: EnvironmentConfig = EnvironmentConfig(),
        sampling_args: dict | None = None,
        rollout_config: RolloutConfig = RolloutConfig(),
        logger: logging.Logger | None = None,
    ):
        """
        Initialize Environment.

        Args:
            csv_path: Path to CSV file
            config: Environment configuration
            sampling_args: Sampling arguments for the model
            logger: Optional logger for output (None = silent execution)
        """
        self.csv_path = csv_path
        self.config = config
        self.rollout_config = rollout_config
        self.model = APILLM(model=config.model, sampling_args=sampling_args or {})
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

    def handle_max_turns_reached(self, state: StateConfig) -> None:
        """Handle reaching max turns: prompt for final output and get response."""
        self.logger.info("max_turns_reached", extra={"max_turns": state.n_turns})

        # Create a special turn for the final message prompt
        final_prompt_turn = Turn(
            turn_number=state.current_turn,
            timestamp=datetime.now(),
            model_response="",  # No model response yet
            truncated_response="",
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
            truncated_response=response,
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
        Process a single turn: extract code, execute, build feedback, check completion.

        Modifies state in-place.

        TODO: Will be rewritten in Phase 2 to extract and execute Python cells.
        """
        # Stub for now - will be implemented in Phase 2
        turn = Turn(
            turn_number=state.current_turn,
            timestamp=datetime.now(),
            model_response=response,
            truncated_response=response,
            done_signal=False,
            feedback_message="TODO: Implement code cell execution",
            reasoning=None,
        )

        state.conversation_manager.add_turn(turn)
        state.is_completed = True  # For now, complete after one turn

    def rollout(self, input: str) -> StateConfig:
        """Execute a multi-turn rollout episode."""
        state = self.init_state(input)

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

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
from src.tools import parse_tool_call, run_tool
from src.types import EnvironmentConfig, StateConfig, Turn, ToolCall, ConversationManager


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
        sampling_args: dict = {},
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
        self.model = APILLM(model=config.model, sampling_args=sampling_args)
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

    def init_state(self, input):
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
            input=input,
            conversation_manager=conversation_manager,
            n_turns=self.config.max_turns,
            is_completed=False,
            current_turn=0,
        )

    def build_system_prompt(self) -> str:
        """Build system prompt from rollout config."""
        return self.rollout_config.system_prompt

    def parse_response_code_blocks(self, response: str) -> tuple[str, list[str]]:
        """Extract code blocks from response and truncate after last </code>."""
        code_blocks = re.findall(r'<code>(.*?)</code>', response, re.DOTALL)

        if code_blocks:
            last_code_end = response.rfind("</code>")
            if last_code_end != -1:
                response = response[:last_code_end + len("</code>")]

        return response, code_blocks

    def check_done_signal(self, response: str, has_code_blocks: bool) -> bool:
        """Check if response contains DONE signal (only valid without tool calls)."""
        return re.search(r"^DONE\b", response, re.MULTILINE) and not has_code_blocks

    def _execute_and_structure_tools(self, code_blocks: list[str]) -> list[ToolCall]:
        """Execute tools and return structured ToolCall objects."""
        tool_calls = []
        for code in code_blocks:
            tool_name, output, success = self._execute_tool_call(code.strip())

            # Parse params from code
            params = {}
            if success:
                try:
                    data = json.loads(code.strip())
                    params = {k: v for k, v in data.items() if k != "tool"}
                except:
                    pass

            tool_call = ToolCall(
                tool_name=tool_name,
                params=params,
                raw_code=code,
                result=output,
                success=success,
                timestamp=datetime.now()
            )
            tool_calls.append(tool_call)

            # Log as before
            self.logger.info("tool_executed", extra={
                "tool": tool_name,
                "output": output,
                "success": success,
            })

        return tool_calls

    def _build_feedback_from_tool_calls(
        self,
        tool_calls: list[ToolCall],
        has_code_blocks: bool
    ) -> str:
        """Build feedback message from structured tool calls."""
        if has_code_blocks:
            results = []
            for i, tc in enumerate(tool_calls, 1):
                results.append(f"[Call {i}]\n[{tc.tool_name}]\n{tc.result}")
            feedback = "\n\n".join(results)
        else:
            self.logger.info("no_tool_call")
            feedback = "No tool call found. Use <code>{\"tool\": \"...\", ...}</code> to explore the data."

        feedback += self.rollout_config.continue_msg
        return feedback

    def handle_max_turns_reached(self, state: StateConfig) -> None:
        """Handle reaching max turns: prompt for final output and get response."""
        self.logger.info("max_turns_reached", extra={"max_turns": state.n_turns})

        # Create a special turn for the final message prompt
        final_prompt_turn = Turn(
            turn_number=state.current_turn,
            timestamp=datetime.now(),
            model_response="",  # No model response yet
            truncated_response="",
            tool_calls=[],
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
            tool_calls=[],
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
        Process a single turn: parse, execute tools, build feedback, check completion.

        Modifies state in-place.
        """
        # 1. Parse code blocks from response
        truncated_response, code_blocks = self.parse_response_code_blocks(response)

        # 2. Execute tools and create structured ToolCall objects
        tool_calls = self._execute_and_structure_tools(code_blocks)

        # 3. Build feedback message
        feedback = self._build_feedback_from_tool_calls(tool_calls, bool(code_blocks))

        # 4. Check for DONE signal (only valid without tool calls)
        done_signal = self.check_done_signal(truncated_response, bool(code_blocks))

        # 5. Create Turn object
        turn = Turn(
            turn_number=state.current_turn,
            timestamp=datetime.now(),
            model_response=response,
            truncated_response=truncated_response,
            tool_calls=tool_calls,
            done_signal=done_signal,
            feedback_message=feedback,
            reasoning=None,  # Future: extract from response
        )

        # 6. Add turn to conversation manager (auto-purges if needed)
        state.conversation_manager.add_turn(turn)

        # 7. Check completion
        if done_signal:
            state.is_completed = True
            self.logger.info("episode_complete", extra={"results": None})

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

            # Process this turn (adds to conversation, executes tools, checks DONE)
            self.process_turn(state, response)

            # Increment turn counter
            state.current_turn += 1

        return state

    def _execute_tool_call(self, code: str) -> tuple[str, str, bool]:
        """
        Parse and execute a tool call.

        Args:
            code: Tool call code string

        Returns:
            Tuple of (tool_name, output, success)
        """
        result = parse_tool_call(code)

        if isinstance(result, str):
            return ("error", result, False)

        tool_name, params = result
        output = run_tool(tool_name, self.df, params)
        return (tool_name, output, True)

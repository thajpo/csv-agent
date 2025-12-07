"""
Environment class for CSV agent.

This is a pure RL-style environment that executes episodes (rollouts)
for CSV exploration and question generation. It uses Python's logging
module for output, keeping the environment logic separate from presentation.
"""

import logging
import re

import pandas as pd

from src.model import APILLM
from src.rich_logger import LogContext
from src.prompts import generate_data_overview, RolloutConfig
from src.tools import parse_tool_call, run_tool
from src.types import EnvironmentConfig, StateConfig


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

        return StateConfig(
            input=input,
            conversation=[{"role": "system", "content": sys_prompt}],
            results_data=[data_overview],
            n_turns=self.config.max_turns,
            is_completed=False,
            current_turn=0,
        )

    def process_model_response(self, state: StateConfig, response: str):
        code_blocks =  re.findall(r'<code>(.*?)</code>', response, re.DOTALL)

        if code_blocks:
            # Find the last </code> tag and truncate everything after it
            last_code_end = response.rfind("</code>")
            if last_code_end != -1:
                truncated = response[: last_code_end + len("</code>")]
                if len(truncated) < len(response):
                    response = truncated

            # Tool calling
            tool_results = []
            for i, code in enumerate(code_blocks, 1):
                tool_name, output, success = self._execute_tool_call(code.strip())
                self.logger.info("tool_executed",extra={"tool": tool_name,"output": output,"success": success,})
                tool_results.append(f"[Call {i}]\n[{tool_name}]\n{output}")
            feedback = "\n\n".join(tool_results)
        else:
            self.logger.info("no_tool_call")
            feedback = "No tool call found. Use <code>{\"tool\": \"...\", ...}</code> to explore the data."

        feedback += self.rollout_config.continue_msg
        state.conversation.append({"role": "user", "content": feedback})

        if re.search(r"^DONE\b", response, re.MULTILINE) and not code_blocks:
            results_data = None
            self.logger.info("episode_complete", extra={"results": results_data})
            state.is_completed = True

        else:
            # Reached max turns without DONE
            self.logger.info("max_turns_reached", extra={"max_turns": state.n_turns})

            state.conversation.append({"role": "user", "content": self.rollout_config.final_msg})

            with LogContext(self.logger, "model_thinking"):
                response = self.model(state.conversation)

            self.logger.info("model_response", extra={"response": response})

            results_data = None

            self.logger.info("episode_complete", extra={"results": results_data})
        return response, tool_results, code_blocks


    def rollout(
        self,
        input,
    ):
        state = self.init_state(input)
        while not state.is_completed:
            self.logger.info("turn_start", extra={"turn": state.current_turn, "max_turns": state.n_turns})

            # Get model response
            with LogContext(self.logger, "model_thinking"):
                response = self.model(state.conversation)

            self.add_model_response(state, response)
            response, tool_calls, code_blocks = self.process_model_response(state, response)

            self.logger.info("model_response", extra={"response": response})

            state.conversation.append({"role": "assistant", "content": response})

            # Check for done signal (only valid if NO tool calls in response)

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

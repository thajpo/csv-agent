"""
Environment class for CSV agent.

This is a pure RL-style environment that executes episodes (rollouts)
for CSV exploration and question generation. It uses Python's logging
module for output, keeping the environment logic separate from presentation.
"""

import logging
import re
from dataclasses import dataclass

import pandas as pd

from src.model import APILLM
from src.rich_logger import LogContext
from src.prompts import generate_data_overview
from src.text_extraction import extract_code_blocks
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

    def rollout(
        self,
        input,
        model,
        sampling_args,
    ):
        state = self.init_state(input)
        while not state.is_completed:

            if self.logger:
                self.logger.info("turn_start", extra={"turn": state.current_turn, "max_turns": state.n_turns})

            # Get model response
            with LogContext(self.logger, "model_thinking"):
                response = self.model(state.conversation)

            self.add_model_response(state, response)

            # Check for tool calls FIRST - if present, truncate response after them
            code_blocks = extract_code_blocks(response)

            if code_blocks:
                # Find the last </code> tag and truncate everything after it
                last_code_end = response.rfind("</code>")
                if last_code_end != -1:
                    truncated = response[: last_code_end + len("</code>")]
                    if len(truncated) < len(response):
                        response = truncated

            if self.logger:
                self.logger.info("model_response", extra={"response": response})

            state.conversation.append({"role": "assistant", "content": response})

            # Check for done signal (only valid if NO tool calls in response)
            if re.search(r"^DONE\b", response, re.MULTILINE) and not code_blocks:
                if extractor:
                    results_data = extractor(response)
                if self.logger:
                    self.logger.info("episode_complete", extra={"results": results_data})
                break

            if not code_blocks:
                if self.logger:
                    self.logger.info("no_tool_call")
                feedback = "No tool call found. Use <code>{\"tool\": \"...\", ...}</code> to explore the data."
            else:
                tool_results = []
                for i, code in enumerate(code_blocks, 1):
                    tool_name, output, success = self._execute_tool_call(code.strip())
                    if self.logger:
                        self.logger.info(
                            "tool_executed",
                            extra={
                                "tool": tool_name,
                                "output": output,
                                "success": success,
                            },
                        )
                    tool_results.append(f"[Call {i}]\n[{tool_name}]\n{output}")

                feedback = "\n\n".join(tool_results)

            feedback += continue_msg
            conversation.append({"role": "user", "content": feedback})

        else:
            # Reached max turns without DONE
            if self.logger:
                self.logger.info("max_turns_reached", extra={"max_turns": max_turns})

            conversation.append({"role": "user", "content": final_msg})

            with LogContext(self.logger, "model_thinking"):
                response = self.model(conversation)

            if self.logger:
                self.logger.info("model_response", extra={"response": response})

            if extractor:
                results_data = extractor(response)

            if self.logger:
                self.logger.info("episode_complete", extra={"results": results_data})

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

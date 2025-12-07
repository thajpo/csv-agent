"""
Test script for the rich logger implementation.

This script demonstrates the hybrid logging approach with Environment + RichHandler.
"""

import logging
from dataclasses import dataclass

from src.environment import Environment
from src.rich_logger import setup_rich_logger
from src.types import EnvironmentConfig


# Mock RolloutConfig for testing
@dataclass
class RolloutConfig:
    """Mock rollout config for testing."""
    mode: str = "explore"
    system_prompt: str = "Test system prompt"
    continue_msg: str = ""
    final_msg: str = ""


def test_silent_execution():
    """Test silent execution (no logger)."""
    print("\n=== TEST 1: Silent Execution (No Logger) ===")

    env = Environment(
        csv_path="data.csv",
        config=EnvironmentConfig(model="grok-4.1-fast", max_turns=1),
        logger=None  # Silent
    )

    # This should run silently with no output
    # (except for any prints from the model itself)
    print("Environment created with logger=None")
    print("This should be silent when rollout() is called")


def test_rich_output():
    """Test with RichHandler."""
    print("\n=== TEST 2: Rich Output ===")

    rollout_config = RolloutConfig()
    logger = setup_rich_logger(rollout_config)

    env = Environment(
        csv_path="data.csv",
        config=EnvironmentConfig(model="grok-4.1-fast", max_turns=1),
        logger=logger
    )

    print("Environment created with RichHandler")
    print("This should produce Rich formatted output when rollout() is called")


def test_file_logging():
    """Test with file logging."""
    print("\n=== TEST 3: File Logging ===")

    logger = logging.getLogger("csv_agent")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler("test_output.log"))

    env = Environment(
        csv_path="data.csv",
        config=EnvironmentConfig(model="grok-4.1-fast", max_turns=1),
        logger=logger
    )

    print("Environment created with FileHandler")
    print("This should write logs to test_output.log")


def test_custom_logging():
    """Test with custom JSON handler."""
    print("\n=== TEST 4: Custom JSON Logging ===")

    import json

    class JSONHandler(logging.Handler):
        def emit(self, record):
            log_entry = {
                "event": record.msg,
                "data": {k: v for k, v in record.__dict__.items()
                        if k not in ['name', 'msg', 'args', 'created', 'filename',
                                     'funcName', 'levelname', 'levelno', 'lineno',
                                     'module', 'msecs', 'pathname', 'process',
                                     'processName', 'relativeCreated', 'thread',
                                     'threadName', 'exc_info', 'exc_text', 'stack_info']}
            }
            print(json.dumps(log_entry, indent=2))

    logger = logging.getLogger("csv_agent")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(JSONHandler())

    env = Environment(
        csv_path="data.csv",
        config=EnvironmentConfig(model="grok-4.1-fast", max_turns=1),
        logger=logger
    )

    print("Environment created with JSONHandler")
    print("This should output structured JSON logs")


def test_manual_logging():
    """Test manual logger calls to verify RichHandler formatting."""
    print("\n=== TEST 5: Manual Logger Calls ===")

    rollout_config = RolloutConfig()
    logger = setup_rich_logger(rollout_config)

    # Manually trigger log events to see RichHandler formatting
    logger.info("episode_start", extra={"csv_path": "test.csv"})
    logger.info("turn_start", extra={"turn": 1, "max_turns": 5})
    logger.info("model_response", extra={"response": "This is a test response from the model."})
    logger.info("tool_executed", extra={"tool": "group_stat", "output": "Mean: 42.5", "success": True})
    logger.info("tool_executed", extra={"tool": "bad_tool", "output": "Error: tool failed", "success": False})
    logger.info("episode_complete", extra={"results": None})

    print("\nAbove should show Rich-formatted output")


if __name__ == "__main__":
    print("Testing Rich Logger Implementation")
    print("=" * 50)

    # Run tests
    test_manual_logging()

    # Uncomment to test other scenarios:
    # test_silent_execution()
    # test_rich_output()
    # test_file_logging()
    # test_custom_logging()

    print("\n" + "=" * 50)
    print("Tests complete!")

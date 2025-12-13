"""
Test script for the rich logger implementation.

This script demonstrates the hybrid logging approach with Environment + RichHandler.
"""

import logging

from src.core.environment import Environment
from src.core.config import DataConfig, ModelConfig, ExecutionConfig, TaskConfig
from src.utils.logger import create_logger


def test_silent_execution():
    """Test silent execution (no logger)."""
    print("\n=== TEST 1: Silent Execution (No Logger) ===")

    env = Environment(
        data=DataConfig(csv_path="csv/data.csv"),
        model=ModelConfig(model_name="grok-4.1-fast"),
        execution=ExecutionConfig(max_turns=1),
        task=TaskConfig(mode="teacher-tutor"),
        logger=create_logger(silent=True)
    )

    # This should run silently with no output
    # (except for any prints from the model itself)
    print("Environment created with silent logger")
    print("This should be silent when rollout() is called")


def test_rich_output():
    """Test with standard logger."""
    print("\n=== TEST 2: Standard Output ===")

    logger = create_logger()

    env = Environment(
        data=DataConfig(csv_path="csv/data.csv"),
        model=ModelConfig(model_name="grok-4.1-fast"),
        execution=ExecutionConfig(max_turns=1),
        task=TaskConfig(mode="teacher-tutor"),
        logger=logger
    )

    print("Environment created with standard logger")
    print("This should produce formatted output when rollout() is called")


def test_file_logging():
    """Test with file logging."""
    print("\n=== TEST 3: File Logging ===")

    logger = logging.getLogger("csv_agent")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler("test_output.log"))

    env = Environment(
        data=DataConfig(csv_path="csv/data.csv"),
        model=ModelConfig(model_name="grok-4.1-fast"),
        execution=ExecutionConfig(max_turns=1),
        task=TaskConfig(mode="teacher-tutor"),
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
        data=DataConfig(csv_path="csv/data.csv"),
        model=ModelConfig(model_name="grok-4.1-fast"),
        execution=ExecutionConfig(max_turns=1),
        task=TaskConfig(mode="teacher-tutor"),
        logger=logger
    )

    print("Environment created with JSONHandler")
    print("This should output structured JSON logs")


def test_manual_logging():
    """Test manual logger calls to verify logger formatting."""
    print("\n=== TEST 5: Manual Logger Calls ===")

    logger = create_logger()

    # Manually trigger log events to see logger formatting
    logger.info("episode_start", extra={"csv_path": "test.csv"})
    logger.info("turn_start", extra={"turn": 1, "max_turns": 5})
    logger.info("model_response", extra={"response": "This is a test response from the model."})
    logger.info("code_executed", extra={"success": True, "stdout": "Mean: 42.5", "stderr": ""})
    logger.info("code_executed", extra={"success": False, "stdout": "", "stderr": "Error: execution failed"})
    logger.info("episode_complete", extra={"results": None})

    print("\nAbove should show formatted output")


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

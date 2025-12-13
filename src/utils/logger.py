"""Simple console logger for CSV agent."""
import logging


def create_logger(name: str = "csv_agent", silent: bool = False) -> logging.Logger:
    """
    Create a basic logger for non-interactive output.

    Args:
        name: Logger name
        silent: If True, use NullHandler (no output)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    if not silent:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(handler)
    else:
        logger.addHandler(logging.NullHandler())

    logger.propagate = False
    return logger

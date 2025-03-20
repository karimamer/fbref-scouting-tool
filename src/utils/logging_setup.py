"""
Logging configuration for the soccer analysis application.
"""
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any

from config.settings import LOGGING


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to the log file
        log_format: Format string for log messages

    Returns:
        Logger instance
    """
    # Use settings from config if not specified
    level = level or LOGGING.get("level", "INFO")
    log_file = log_file or LOGGING.get("file", "soccer_analysis.log")
    log_format = log_format or LOGGING.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    # Create logger
    logger = logging.getLogger("soccer_analysis")
    logger.setLevel(numeric_level)

    # Create formatters and handlers
    formatter = logging.Formatter(log_format)

    # Create directory for log file if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log setup complete
    logger.info(f"Logging setup complete with level {level}")

    return logger


def log_execution_time(logger: logging.Logger, start_time: datetime, operation: str) -> None:
    """
    Log the execution time of an operation.

    Args:
        logger: Logger instance
        start_time: Start time of the operation
        operation: Description of the operation
    """
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"{operation} completed in {duration.total_seconds():.2f} seconds")


def log_data_stats(logger: logging.Logger, df: Any, name: str) -> None:
    """
    Log basic statistics about a DataFrame.

    Args:
        logger: Logger instance
        df: DataFrame to log statistics for
        name: Name of the DataFrame
    """
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            logger.info(f"{name} shape: {df.shape}, columns: {len(df.columns)}")
            if not df.empty:
                logger.debug(f"{name} columns: {', '.join(df.columns)}")
        else:
            logger.info(f"{name} is not a DataFrame")
    except Exception as e:
        logger.warning(f"Error logging DataFrame stats: {str(e)}")

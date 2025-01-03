# utils/logging_utils.py
import logging
import os
from pathlib import Path

_logger = None

def setup_logger(output_dir: str, logger_name: str = "shared_logger") -> logging.Logger:
    """
    Sets up a logger that writes to a file (run.log) and to console.
    Returns a logger instance.
    """
    global _logger
    if _logger is not None:
        return _logger  # Return the existing logger if already created

    # Make sure the output directory exists
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Log file path
    log_file = output_dir_path / "run.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # File handler
    fh = logging.FileHandler(str(log_file))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    _logger = logger  # Store the logger in the global variable
    return logger

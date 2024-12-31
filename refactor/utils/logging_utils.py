import logging
import os
from datetime import datetime
from pathlib import Path

_logger = None 

def setup_logger(output_dir: str, logger_name: str = "shared_logger") -> logging.Logger:
    """
    Sets up a logger that writes to both a timestamped file and to console.
    Returns a logger instance.
    """

    global _logger
    if _logger is not None:
        return _logger  # Return the existing logger

    # Create a new logger instance
    timestamp_str = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    run_output_dir = Path(output_dir) / timestamp_str
    run_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_output_dir / "run.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    logger.propagate = False

    # File handler
    fh = logging.FileHandler(str(log_file))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Optional console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    _logger = logger  # Store the logger in the global variable
    return logger

import logging
import sys


def setup_logger(name, level=logging.DEBUG):
    """Set up a console-only logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Avoid adding handlers multiple times
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.propagate = False

    return logger

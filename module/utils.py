import logging
import sys

DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def setup_logging(level: int = DEFAULT_LOG_LEVEL, log_format: str = LOG_FORMAT) -> None:
    """
    Configures the root logger for the application.

    Args:
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        log_format (str): The format string for log messages.
    """
    # Get the root logger
    root_logger = logging.getLogger()

    # Remove any existing handlers to avoid duplicate messages
    # if root_logger.hasHandlers():
    #     root_logger.handlers.clear() # Be cautious with this in complex setups or libraries

    # Configure the root logger - if it's already configured, this might not reconfigure it
    # unless force=True is used (Python 3.8+) or handlers are cleared.
    # For simplicity and broad compatibility, we'll set the level and add a handler if none exist for our format.
    root_logger.setLevel(level)

    # Add a stream handler if no appropriate handlers are configured
    if not any(isinstance(h, logging.StreamHandler) and h.formatter and h.formatter._fmt == log_format for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level) # Ensure handler also respects the level
        root_logger.addHandler(console_handler)
        logging.info("Root logger configured by setup_logging.")
    elif not root_logger.handlers: # If no handlers at all, add one.
        # This case might be redundant if the above check is comprehensive
        # but acts as a fallback.
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)
        logging.info("Root logger (no prior handlers) configured by setup_logging.")
    else:
        # Logger might already be configured, perhaps by an import or another part of the system.
        # We'll just ensure the level is set.
        logging.info(f"Root logger level set to {logging.getLevelName(level)}. Existing handlers detected.")

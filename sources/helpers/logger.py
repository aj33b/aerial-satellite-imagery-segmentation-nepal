import os
import logging

from logging.handlers import TimedRotatingFileHandler
from tqdm import tqdm


class DynamicTqdmHandler(logging.StreamHandler):
    """
    Custom logging handler that dynamically switches between `tqdm.write` and standard `StreamHandler`
    based on whether a `tqdm` progress bar is currently active.
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            # If tqdm is active, use tqdm.write to log below the progress bar
            if tqdm.get_lock().locks:  # Check if a tqdm instance is active
                tqdm.write(msg)
            else:
                # If tqdm is not active, fallback to standard `StreamHandler` behavior
                super().emit(record)
        except Exception:
            self.handleError(record)

class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[1;91m"  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        formatted_message = super().format(record)
        return f"{color}{formatted_message}{self.RESET}"

class LoggerHelper:
    def __init__(self,logger_name="Satellite Segmentation Nepal",
                     log_level=logging.DEBUG,
                     log_dir=r"../logs"):
        self.logger_name = logger_name
        self.log_level = log_level
        self.log_dir = log_dir
        self.logger = self.__setup_logger()

    def __setup_logger(self):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.log_level)

        # Avoid duplicate handlers in case the logger is already initialized
        if logger.handlers:
            return logger

        # Create the logs directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Log format
        LOG_FORMAT = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s "
            "[in %(pathname)s:%(lineno)d]"
        )

        # Console handler with dynamic tqdm support
        stream_handler = DynamicTqdmHandler()
        stream_handler.setLevel(self.log_level)
        stream_handler.setFormatter(ColorFormatter(LOG_FORMAT))
        logger.addHandler(stream_handler)

        # Timed rotating file handler (rotates logs daily at midnight)
        file_handler = TimedRotatingFileHandler(
            os.path.join(self.log_dir, "log.txt"),
            when="midnight",
            interval=1,
            backupCount=7,  # Keep logs for the last 7 days
            encoding="utf-8"
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)

        return logger
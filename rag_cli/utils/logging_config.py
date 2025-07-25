"""Logging configuration for RAG CLI"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler

from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# Custom theme for rich logging
LOGGING_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "critical": "bold white on red",
        "debug": "dim cyan",
        "success": "bold green",
    }
)


class LoggingConfig:
    """Manages logging configuration for the application"""

    # Log levels
    LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    @staticmethod
    def setup_logging(
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        console_output: bool = True,
        rich_formatting: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
    ) -> None:
        """
        Setup comprehensive logging configuration

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
            console_output: Whether to output to console
            rich_formatting: Whether to use rich formatting for console
            max_file_size: Maximum size of log file before rotation
            backup_count: Number of backup files to keep
        """
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(LoggingConfig.LEVELS.get(log_level.upper(), logging.INFO))

        # Clear existing handlers
        root_logger.handlers.clear()

        # Console handler
        if console_output:
            if rich_formatting:
                console_handler = RichHandler(
                    console=Console(theme=LOGGING_THEME),
                    show_path=False,
                    enable_link_path=True,
                )
                console_handler.setFormatter(logging.Formatter("%(message)s"))
            else:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                )

            root_logger.addHandler(console_handler)

        # File handler
        if log_file:
            # Create log directory if needed
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Rotating file handler
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_file_size, backupCount=backup_count
            )

            # Detailed format for file logs
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            )
            file_handler.setFormatter(file_formatter)

            root_logger.addHandler(file_handler)

        # Configure specific loggers
        LoggingConfig._configure_third_party_loggers()

        # Log initial setup
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured: level={log_level}, file={log_file}")

    @staticmethod
    def _configure_third_party_loggers() -> None:
        """Configure third-party library loggers"""
        # Reduce noise from third-party libraries
        noisy_loggers = [
            "urllib3",
            "asyncio",
            "chromadb",
            "sentence_transformers",
            "transformers",
            "huggingface_hub",
        ]

        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get a logger instance

        Args:
            name: Logger name (usually __name__)

        Returns:
            Logger instance
        """
        return logging.getLogger(name)

    @staticmethod
    def setup_error_logging(error_log_file: str = "errors.log") -> None:
        """
        Setup separate error logging

        Args:
            error_log_file: Path to error log file
        """
        error_logger = logging.getLogger("errors")
        error_logger.setLevel(logging.ERROR)

        # Create error file handler
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
        )

        # Detailed error format
        error_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s\n"
            "Function: %(funcName)s - Line: %(lineno)d\n"
            "Message: %(message)s\n"
            "---"
        )
        error_handler.setFormatter(error_formatter)

        error_logger.addHandler(error_handler)

    @staticmethod
    def log_exception(
        logger: logging.Logger, exception: Exception, message: str = "An error occurred"
    ) -> None:
        """
        Log an exception with full traceback

        Args:
            logger: Logger instance
            exception: Exception to log
            message: Error message
        """
        from rag_cli.utils.error_utils import ErrorUtils

        formatted_msg = ErrorUtils.format_error_message(message, exception)
        logger.error(formatted_msg, exc_info=True)

    @staticmethod
    def create_session_logger(
        session_id: str, log_dir: str = "logs/sessions"
    ) -> logging.Logger:
        """
        Create a logger for a specific session

        Args:
            session_id: Unique session identifier
            log_dir: Directory for session logs

        Returns:
            Session-specific logger
        """
        # Create session logger
        logger = logging.getLogger(f"session.{session_id}")
        logger.setLevel(logging.DEBUG)

        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Session log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_file = log_path / f"session_{session_id}_{timestamp}.log"

        # File handler for session
        handler = logging.FileHandler(session_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        logger.addHandler(handler)
        logger.propagate = False  # Don't propagate to root logger

        return logger


class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        if not hasattr(self, "_logger"):
            self._logger = logging.getLogger(
                self.__class__.__module__ + "." + self.__class__.__name__
            )
        return self._logger

    def log_debug(self, message: str, **kwargs) -> None:
        """Log debug message"""
        self.logger.debug(message, **kwargs)

    def log_info(self, message: str, **kwargs) -> None:
        """Log info message"""
        self.logger.info(message, **kwargs)

    def log_warning(self, message: str, **kwargs) -> None:
        """Log warning message"""
        self.logger.warning(message, **kwargs)

    def log_error(
        self, message: str, exception: Optional[Exception] = None, **kwargs
    ) -> None:
        """Log error message"""
        from rag_cli.utils.error_utils import ErrorUtils

        formatted_msg = ErrorUtils.format_error_message(message, exception)
        self.logger.error(formatted_msg, exc_info=bool(exception), **kwargs)

    def log_critical(self, message: str, **kwargs) -> None:
        """Log critical message"""
        self.logger.critical(message, **kwargs)

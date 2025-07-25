"""Centralized error handling utilities."""

from typing import Dict, Optional, Any


class ErrorUtils:
    """Utility class for common error handling patterns."""

    @staticmethod
    def format_error_message(
        message: str, exception: Optional[Exception] = None
    ) -> str:
        """
        Format error message with exception details.

        Args:
            message: Base error message
            exception: Optional exception to include

        Returns:
            Formatted error message
        """
        if exception:
            return f"{message}: {type(exception).__name__}: {str(exception)}"
        return message

    @staticmethod
    def handle_file_descriptor_error(logger=None) -> Dict[str, Any]:
        """
        Handle file descriptor errors with consistent messaging.

        Args:
            logger: Optional logger instance

        Returns:
            Dictionary with error details and suggestions
        """
        if logger:
            logger.warning("Falling back due to file descriptor error")

        return {
            "error_type": "file_descriptor",
            "message": "System resource limit exceeded",
            "suggestion": (
                "Try one of the following:\n"
                "1. Reduce batch size in settings\n"
                "2. Close other applications\n"
                "3. Run 'ulimit -n 8192' in terminal before starting"
            ),
        }

    @staticmethod
    def handle_connection_error(provider: str) -> Dict[str, str]:
        """
        Get provider-specific connection error messages.

        Args:
            provider: LLM provider name

        Returns:
            Dictionary with error message and suggestions
        """
        messages = {
            "ollama": {
                "message": "Cannot connect to Ollama",
                "suggestion": (
                    "You can:\n"
                    "1. Start Ollama (run 'ollama serve' in terminal)\n"
                    "2. Install the model (run 'ollama pull llama3.2')\n"
                    "3. Or switch to an API provider in Settings (Ctrl+S)"
                ),
            },
            "openai": {
                "message": "Cannot connect to OpenAI API",
                "suggestion": (
                    "Please check:\n"
                    "1. Your API key is correct in Settings (Ctrl+S)\n"
                    "2. Your internet connection is working\n"
                    "3. The API endpoint URL is correct (if using custom endpoint)\n"
                    "4. Or switch to Ollama in Settings (Ctrl+S)"
                ),
            },
            "anthropic": {
                "message": "Cannot connect to Anthropic API",
                "suggestion": (
                    "Please check:\n"
                    "1. Your API key is correct in Settings (Ctrl+S)\n"
                    "2. Your internet connection is working\n"
                    "3. Or switch to Ollama in Settings (Ctrl+S)"
                ),
            },
        }

        default = {
            "message": "Connection error",
            "suggestion": (
                "Please check your network connection and try again.\n"
                "You can also switch providers in Settings (Ctrl+S)"
            ),
        }

        return messages.get(provider, default)

    @staticmethod
    def handle_database_error(error: Exception, db_path: str) -> Dict[str, str]:
        """
        Handle database-related errors.

        Args:
            error: The exception that occurred
            db_path: Database path

        Returns:
            Dictionary with error details and recovery suggestions
        """
        error_str = str(error).lower()

        if "no such table" in error_str:
            return {
                "message": "Database is corrupted or incomplete",
                "suggestion": f"Please delete {db_path} and recreate the database",
            }

        if "permission" in error_str:
            return {
                "message": "Permission denied accessing database",
                "suggestion": f"Check permissions on {db_path} or run with appropriate privileges",
            }

        if "disk full" in error_str or "no space" in error_str:
            return {
                "message": "Insufficient disk space",
                "suggestion": "Free up disk space and try again",
            }

        # Generic database error
        return {
            "message": f"Database error: {str(error)}",
            "suggestion": "Try recreating the database or check the logs for details",
        }

    @staticmethod
    def log_error_with_context(
        logger,
        message: str,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log error with additional context information.

        Args:
            logger: Logger instance
            message: Error message
            exception: Exception that occurred
            context: Optional context dictionary
        """
        error_msg = ErrorUtils.format_error_message(message, exception)

        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            error_msg += f" | Context: {context_str}"

        logger.error(error_msg, exc_info=True)

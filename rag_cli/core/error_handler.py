"""Error handling utilities for RAG system"""

import traceback
from typing import Dict, Any, Optional
from enum import Enum

from rag_cli.utils.logger import RichLogger


class ErrorType(Enum):
    """Types of errors that can occur in the RAG system"""
    CONNECTION_ERROR = "connection_error"
    FILE_DESCRIPTOR_ERROR = "file_descriptor_error"
    MODEL_NOT_FOUND = "model_not_found"
    DATABASE_ERROR = "database_error"
    EMBEDDING_ERROR = "embedding_error"
    QUERY_ERROR = "query_error"
    INITIALIZATION_ERROR = "initialization_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorHandler:
    """Handles errors in the RAG system with helpful messages and recovery suggestions"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or "llama3.2"
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle an error and return structured error information
        
        Args:
            error: The exception that occurred
            
        Returns:
            Dictionary with error details and suggestions
        """
        error_str = str(error)
        error_type = self._classify_error(error, error_str)
        
        # Get specific handler based on error type
        handler_map = {
            ErrorType.CONNECTION_ERROR: self._handle_connection_error,
            ErrorType.FILE_DESCRIPTOR_ERROR: self._handle_file_descriptor_error,
            ErrorType.MODEL_NOT_FOUND: self._handle_model_not_found_error,
            ErrorType.DATABASE_ERROR: self._handle_database_error,
            ErrorType.EMBEDDING_ERROR: self._handle_embedding_error,
            ErrorType.QUERY_ERROR: self._handle_query_error,
            ErrorType.INITIALIZATION_ERROR: self._handle_initialization_error,
            ErrorType.UNKNOWN_ERROR: self._handle_unknown_error
        }
        
        handler = handler_map.get(error_type, self._handle_unknown_error)
        result = handler(error, error_str)
        
        # Add common fields
        result.update({
            "error_type": error_type.value,
            "error_class": type(error).__name__,
            "traceback": traceback.format_exc() if RichLogger.debug_mode else None
        })
        
        # Log the error
        RichLogger.error(f"{error_type.value}: {error_str}")
        
        return result
    
    def _classify_error(self, error: Exception, error_str: str) -> ErrorType:
        """Classify the error type based on the exception and message"""
        
        # Connection errors
        if any(term in error_str.lower() for term in ["connection", "refused", "ollama"]):
            return ErrorType.CONNECTION_ERROR
        
        # File descriptor errors
        if "fds_to_keep" in error_str or "file descriptor" in error_str.lower():
            return ErrorType.FILE_DESCRIPTOR_ERROR
        
        # Model not found
        if "model" in error_str.lower() and any(term in error_str.lower() for term in ["not found", "does not exist", "pull"]):
            return ErrorType.MODEL_NOT_FOUND
        
        # Database errors
        if any(term in error_str.lower() for term in ["database", "vectorstore", "chroma"]):
            return ErrorType.DATABASE_ERROR
        
        # Embedding errors
        if any(term in error_str.lower() for term in ["embedding", "huggingface", "sentence-transformer"]):
            return ErrorType.EMBEDDING_ERROR
        
        # Query errors
        if "query" in error_str.lower():
            return ErrorType.QUERY_ERROR
        
        # Initialization errors
        if "initialization" in error_str.lower() or "initialize" in error_str.lower():
            return ErrorType.INITIALIZATION_ERROR
        
        return ErrorType.UNKNOWN_ERROR
    
    def _handle_connection_error(self, error: Exception, error_str: str) -> Dict[str, Any]:
        """Handle connection errors (usually Ollama not running)"""
        return {
            "message": "Cannot connect to Ollama service",
            "user_message": "The AI model service (Ollama) is not running.",
            "suggestions": [
                "Start Ollama by running: ollama serve",
                f"Check if the model '{self.model_name}' is installed: ollama list",
                f"If not installed, pull it: ollama pull {self.model_name}",
                "Make sure Ollama is running on the default port (11434)"
            ],
            "recovery_action": "start_ollama"
        }
    
    def _handle_file_descriptor_error(self, error: Exception, error_str: str) -> Dict[str, Any]:
        """Handle file descriptor limit errors"""
        return {
            "message": "File descriptor limit exceeded",
            "user_message": "The system has run out of file handles.",
            "suggestions": [
                "Increase the file descriptor limit: ulimit -n 8192",
                "Restart the application after increasing the limit",
                "On macOS, you may need to adjust system limits",
                "Consider closing other applications to free resources"
            ],
            "recovery_action": "increase_fd_limit"
        }
    
    def _handle_model_not_found_error(self, error: Exception, error_str: str) -> Dict[str, Any]:
        """Handle model not found errors"""
        return {
            "message": f"Model '{self.model_name}' not found",
            "user_message": f"The AI model '{self.model_name}' is not installed.",
            "suggestions": [
                f"Install the model: ollama pull {self.model_name}",
                "Check available models: ollama list",
                "Make sure Ollama is running: ollama serve",
                "Try a different model in settings"
            ],
            "recovery_action": "pull_model"
        }
    
    def _handle_database_error(self, error: Exception, error_str: str) -> Dict[str, Any]:
        """Handle database-related errors"""
        return {
            "message": "Database operation failed",
            "user_message": "There was an error with the document database.",
            "suggestions": [
                "Try loading the database again",
                "Check if the database path exists and is accessible",
                "Recreate the database from documents if needed",
                "Check file permissions on the database directory"
            ],
            "recovery_action": "reload_database"
        }
    
    def _handle_embedding_error(self, error: Exception, error_str: str) -> Dict[str, Any]:
        """Handle embedding model errors"""
        return {
            "message": "Embedding model error",
            "user_message": "There was an error loading the text embedding model.",
            "suggestions": [
                "Clear the HuggingFace cache: rm -rf ~/.cache/huggingface",
                "Check your internet connection for model downloads",
                "Try restarting the application",
                "Set TOKENIZERS_PARALLELISM=false environment variable"
            ],
            "recovery_action": "clear_embedding_cache"
        }
    
    def _handle_query_error(self, error: Exception, error_str: str) -> Dict[str, Any]:
        """Handle query processing errors"""
        return {
            "message": "Query processing failed",
            "user_message": "There was an error processing your question.",
            "suggestions": [
                "Try rephrasing your question",
                "Check if the database is loaded",
                "Try without RAG mode if retrieval is failing",
                "Check the application logs for details"
            ],
            "recovery_action": "retry_query"
        }
    
    def _handle_initialization_error(self, error: Exception, error_str: str) -> Dict[str, Any]:
        """Handle initialization errors"""
        return {
            "message": "System initialization failed",
            "user_message": "The RAG system could not be initialized properly.",
            "suggestions": [
                "Check all required services are running (Ollama)",
                "Verify settings are correct",
                "Try resetting to default settings",
                "Check system resources (memory, disk space)"
            ],
            "recovery_action": "reinitialize"
        }
    
    def _handle_unknown_error(self, error: Exception, error_str: str) -> Dict[str, Any]:
        """Handle unknown/uncategorized errors"""
        return {
            "message": "An unexpected error occurred",
            "user_message": "Something went wrong. Please check the logs for details.",
            "suggestions": [
                "Check the application logs for more details",
                "Try restarting the application",
                "Report the issue if it persists",
                f"Error details: {error_str[:200]}"
            ],
            "recovery_action": "restart"
        }
    
    @staticmethod
    def format_error_for_user(error_info: Dict[str, Any]) -> str:
        """
        Format error information for display to the user
        
        Args:
            error_info: Error information dictionary
            
        Returns:
            Formatted error message
        """
        lines = [
            f"❌ {error_info['user_message']}",
            "",
            "Suggestions:",
        ]
        
        for i, suggestion in enumerate(error_info.get('suggestions', []), 1):
            lines.append(f"{i}. {suggestion}")
        
        return "\n".join(lines)
    
    @staticmethod
    def get_recovery_command(recovery_action: str, context: Dict[str, Any] = None) -> Optional[str]:
        """
        Get a command that can help recover from the error
        
        Args:
            recovery_action: The recovery action identifier
            context: Additional context for the recovery
            
        Returns:
            Command string or None
        """
        commands = {
            "start_ollama": "ollama serve",
            "increase_fd_limit": "ulimit -n 8192",
            "pull_model": f"ollama pull {context.get('model_name', 'llama3.2')}" if context else "ollama pull llama3.2",
            "clear_embedding_cache": "rm -rf ~/.cache/huggingface",
            "reinitialize": "# Restart the application",
            "restart": "# Restart the application"
        }
        
        return commands.get(recovery_action)
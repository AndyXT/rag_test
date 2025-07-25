"""Configuration module for RAG CLI"""

# Re-export commonly used constants for backward compatibility
from .model_config import DEFAULT_MODEL as DEFAULT_MODEL
from .model_config import DEFAULT_TEMPERATURE as DEFAULT_TEMPERATURE
from .system_config import DEFAULT_CHUNK_SIZE as DEFAULT_CHUNK_SIZE
from .system_config import DEFAULT_CHUNK_OVERLAP as DEFAULT_CHUNK_OVERLAP
from .database_config import DEFAULT_PERSIST_DIRECTORY as DEFAULT_PERSIST_DIRECTORY

# Export all configuration modules for structured access
from . import model_config, system_config, database_config, ui_config

__all__ = [
    # Re-exported constants
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_PERSIST_DIRECTORY",
    # Configuration modules
    "model_config",
    "system_config",
    "database_config",
    "ui_config",
]

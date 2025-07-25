"""Configuration module for RAG CLI"""

from .model_config import *
from .system_config import *
from .database_config import *
from .ui_config import *

# Re-export commonly used constants for backward compatibility
from .model_config import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from .system_config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from .database_config import DEFAULT_PERSIST_DIRECTORY
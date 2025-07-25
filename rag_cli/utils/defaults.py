"""Default configuration values for RAG CLI application.

This module provides backward compatibility by re-exporting constants
from the new config module structure.
"""

# Import from new config modules
from rag_cli.config.model_config import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_QUERY_EXPANSION_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    LARGE_MODELS
)

from rag_cli.config.system_config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_RETRIEVAL_K,
    DEFAULT_RERANKER_TOP_K,
    DEFAULT_EXPANSION_QUERIES,
    DEFAULT_BATCH_SIZE
)

from rag_cli.config.database_config import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    DEFAULT_DOCUMENTS_DIRECTORY
)

# Additional re-exports for convenience
DEFAULT_MODEL = DEFAULT_OLLAMA_MODEL  # Alias for backward compatibility

__all__ = [
    # Model defaults
    "DEFAULT_MODEL",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_RERANKER_MODEL",
    "DEFAULT_QUERY_EXPANSION_MODEL",
    # LLM parameters
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT",
    # Chunk settings
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    # Retrieval settings
    "DEFAULT_RETRIEVAL_K",
    "DEFAULT_RERANKER_TOP_K",
    "DEFAULT_EXPANSION_QUERIES",
    # Model lists
    "LARGE_MODELS",
    # Database settings
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_PERSIST_DIRECTORY",
    # Batch processing
    "DEFAULT_BATCH_SIZE",
    # File settings
    "DEFAULT_DOCUMENTS_DIRECTORY"
]
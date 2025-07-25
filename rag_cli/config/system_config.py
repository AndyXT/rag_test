"""System-level configuration constants"""

import os
from pathlib import Path

# Chunk settings for document processing
CHUNK_SETTINGS = {
    "default": {"size": 1000, "overlap": 200},
    "small": {"size": 500, "overlap": 100},
    "medium": {"size": 1000, "overlap": 200},
    "large": {"size": 2000, "overlap": 400},
    "code": {"size": 1500, "overlap": 300},
}

DEFAULT_CHUNK_SIZE = CHUNK_SETTINGS["default"]["size"]
DEFAULT_CHUNK_OVERLAP = CHUNK_SETTINGS["default"]["overlap"]

# Retrieval settings
RETRIEVAL_SETTINGS = {
    "k": {
        "default": 3,
        "minimal": 1,
        "standard": 3,
        "comprehensive": 5,
        "exhaustive": 10,
    },
    "reranker_top_k": {"default": 3, "minimal": 1, "standard": 3, "comprehensive": 5},
    "expansion_queries": {
        "default": 3,
        "minimal": 1,
        "standard": 3,
        "comprehensive": 5,
    },
}

DEFAULT_RETRIEVAL_K = RETRIEVAL_SETTINGS["k"]["default"]
DEFAULT_RERANKER_TOP_K = RETRIEVAL_SETTINGS["reranker_top_k"]["default"]
DEFAULT_EXPANSION_QUERIES = RETRIEVAL_SETTINGS["expansion_queries"]["default"]

# System paths
BASE_DIR = Path.home() / ".rag_cli"
CACHE_DIR = BASE_DIR / "cache"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

# Ensure directories exist
for directory in [BASE_DIR, CACHE_DIR, LOGS_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File settings
FILE_SETTINGS = {
    "settings_file": "settings.json",
    "chat_history_file": "chat_history.json",
    "system_prompt_file": "system_prompt.md",
    "log_file": str(LOGS_DIR / "rag_cli.log"),
    "error_log_file": str(LOGS_DIR / "errors.log"),
}

# Resource limits
RESOURCE_LIMITS = {
    "max_file_descriptors": 8192,
    "max_memory_mb": 4096,
    "max_threads": 8,
    "max_workers": 4,
}

# Timeouts (in seconds)
TIMEOUTS = {
    "llm_default": 30,
    "llm_extended": 60,
    "embedding": 30,
    "reranking": 15,
    "database_operation": 120,
    "file_operation": 10,
}

# Batch processing
BATCH_SETTINGS = {
    "default_size": 100,
    "small": 50,
    "medium": 100,
    "large": 200,
    "max_retries": 3,
    "retry_delay": 1.0,
    "backoff_factor": 2.0,
}

DEFAULT_BATCH_SIZE = BATCH_SETTINGS["default_size"]

# Environment variables
ENV_VARS = {
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    "OMP_NUM_THREADS": "1",
    "ANONYMIZED_TELEMETRY": "False",
}

# Apply environment variables
for key, value in ENV_VARS.items():
    os.environ[key] = value

# Feature flags
FEATURES = {
    "query_expansion": True,
    "reranking": True,
    "caching": True,
    "streaming": True,
    "async_processing": True,
    "auto_retry": True,
    "progress_tracking": True,
}

# Debug settings
DEBUG_SETTINGS = {
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "log_to_file": True,
    "log_to_console": True,
    "show_timings": False,
    "show_memory_usage": False,
    "trace_calls": False,
}

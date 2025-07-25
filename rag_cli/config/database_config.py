"""Database-related configuration constants"""

from pathlib import Path

# ChromaDB settings
CHROMA_SETTINGS = {
    "persist_directory": "chroma_db",
    "collection_name": "rag_docs",
    "distance_metric": "cosine",  # cosine, l2, ip (inner product)
    "anonymized_telemetry": False
}

DEFAULT_PERSIST_DIRECTORY = CHROMA_SETTINGS["persist_directory"]
DEFAULT_COLLECTION_NAME = CHROMA_SETTINGS["collection_name"]

# Database paths
DATABASE_PATHS = {
    "default": Path("./chroma_db"),
    "user": Path.home() / ".rag_cli" / "databases" / "default",
    "shared": Path("/usr/share/rag_cli/databases"),
    "temp": Path("/tmp/rag_cli_db")
}

DEFAULT_DOCUMENTS_DIRECTORY = "documents"

# Vector database settings
VECTOR_DB_SETTINGS = {
    "similarity_threshold": 0.7,
    "max_results": 100,
    "index_params": {
        "efConstruction": 200,
        "M": 16
    },
    "search_params": {
        "ef": 50
    }
}

# Backup settings
BACKUP_SETTINGS = {
    "enabled": True,
    "max_backups": 3,
    "backup_on_create": True,
    "backup_on_major_change": True,
    "compression": True
}

# Database maintenance
MAINTENANCE_SETTINGS = {
    "auto_optimize": True,
    "optimize_threshold_mb": 1000,
    "vacuum_on_startup": False,
    "check_integrity_on_load": True
}

# Collection metadata
DEFAULT_COLLECTION_METADATA = {
    "created_by": "rag_cli",
    "version": "1.0",
    "description": "Document collection for RAG system"
}

# Supported document formats
SUPPORTED_FORMATS = {
    "documents": [".pdf", ".txt", ".md", ".docx"],
    "code": [".py", ".js", ".ts", ".java", ".cpp", ".c", ".rs", ".go"],
    "data": [".json", ".yaml", ".yml", ".csv", ".xml"],
    "web": [".html", ".htm"]
}

# File processing settings
FILE_PROCESSING = {
    "max_file_size_mb": 100,
    "encoding": "utf-8",
    "extract_metadata": True,
    "clean_text": True,
    "remove_headers_footers": True
}

# Index settings
INDEX_SETTINGS = {
    "build_async": True,
    "update_async": True,
    "merge_async": True,
    "persist_every_n_documents": 1000
}

# Cache settings for embeddings
EMBEDDING_CACHE = {
    "enabled": True,
    "cache_dir": Path.home() / ".cache" / "rag_cli" / "embeddings",
    "max_cache_size_gb": 5,
    "ttl_days": 30,
    "cleanup_on_startup": True
}
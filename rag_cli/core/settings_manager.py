"""Settings manager for RAG CLI application"""

import os
import json
from typing import Dict, Any

from rag_cli.utils.logger import RichLogger


class SettingsManager:
    """Manages application settings with persistence"""

    DEFAULT_SETTINGS_FILE = "settings.json"

    def __init__(self, settings_file: str = DEFAULT_SETTINGS_FILE):
        self.settings_file = settings_file
        self._file_exists_cache = None  # Cache file existence check
        self._last_mtime = None  # Cache file modification time
        self.default_settings = {
            "ollama_model": "llama3.2:3b",
            "temperature": 0.1,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "retrieval_k": 3,
            "auto_save": True,
            "dark_mode": False,
            "show_context": False,  # Toggle for showing retrieved context
            "use_reranker": False,  # Toggle for reranking
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Reranker model (smaller, more accessible)
            "reranker_top_k": 3,  # Number of documents to keep after reranking
            "use_query_expansion": False,  # Toggle for query expansion
            "query_expansion_model": "llama3.2:3b",  # Small fast model for query expansion
            "expansion_queries": 3,  # Number of expanded queries to generate
            "llm_provider": "ollama",  # 'ollama', 'openai', 'anthropic'
            "api_key": "",  # For API providers
            "api_base_url": "",  # For custom API endpoints
            "openai_model": "gpt-3.5-turbo",  # For OpenAI
            "anthropic_model": "claude-3-haiku-20240307",  # For Anthropic
        }
        self.settings = self.load_settings()

    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file or return defaults with caching"""
        try:
            # Check cache first
            if self._file_exists_cache is None:
                self._file_exists_cache = os.path.exists(self.settings_file)
            
            if self._file_exists_cache:
                # Check if file has been modified
                current_mtime = os.path.getmtime(self.settings_file)
                if self._last_mtime != current_mtime:
                    with open(self.settings_file, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                        # Update cache
                        self._last_mtime = current_mtime
                        # Merge with defaults to handle missing keys
                        return {**self.default_settings, **loaded}
                # If file hasn't changed and we have settings, return them
                elif hasattr(self, 'settings') and self.settings:
                    return self.settings
        except (
            IOError,
            OSError,
            PermissionError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            AttributeError,
        ) as e:
            RichLogger.warning(
                f"Could not load settings from {self.settings_file}: {e}"
            )
            # Invalidate cache on error
            self._file_exists_cache = False
            self._last_mtime = None
        return self.default_settings.copy()

    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """Save settings to file"""
        try:
            # Merge with current settings
            self.settings.update(settings)
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
            return True
        except (IOError, OSError, PermissionError, TypeError) as e:
            RichLogger.warning(f"Could not save settings to {self.settings_file}: {e}")
            return False

    def get(self, key: str, default=None) -> Any:
        """Get a setting value"""
        return self.settings.get(key, default)

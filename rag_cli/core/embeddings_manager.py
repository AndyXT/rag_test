"""Embeddings manager for handling HuggingFace embeddings initialization and management"""

import os
import gc
import shutil
from pathlib import Path
from typing import Optional, Any
import time

from langchain_huggingface import HuggingFaceEmbeddings

from rag_cli.utils.logger import RichLogger
from rag_cli.utils.defaults import DEFAULT_EMBEDDING_MODEL


class EmbeddingsManager:
    """Manages embeddings initialization with cache management"""
    
    def __init__(self, settings_manager: Optional[Any] = None):
        self.settings_manager = settings_manager
        self.embeddings = None
        self._embedding_model = None
    
    def initialize(self, model_name: Optional[str] = None):
        """Initialize embeddings with cache management"""
        # Use provided model or get from settings
        if model_name:
            self._embedding_model = model_name
        else:
            self._embedding_model = (
                self.settings_manager.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
                if self.settings_manager
                else DEFAULT_EMBEDDING_MODEL
            )
        
        # Clean cache before initialization
        self._clean_hf_cache_locks()
        
        # Initialize embeddings
        self._initialize_embeddings_safely()
    
    def _clean_hf_cache_locks(self):
        """Clean HuggingFace cache lock files more aggressively"""
        try:
            cache_dir = Path.home() / ".cache" / "huggingface"
            if not cache_dir.exists():
                return
                
            cleaned_files = []
            
            # Clean lock files and temp files
            for pattern in ["*.lock", "*.tmp*", "tmp_*", "*~"]:
                for lock_file in cache_dir.rglob(pattern):
                    try:
                        # Remove any lock files older than 5 minutes (more aggressive)
                        if lock_file.stat().st_mtime < (time.time() - 300):
                            lock_file.unlink()
                            cleaned_files.append(lock_file.name)
                    except Exception:
                        pass  # Non-critical - lock file may be in use
            
            # Clean zero-byte files
            for file_path in cache_dir.rglob("*"):
                if file_path.is_file() and file_path.stat().st_size == 0:
                    try:
                        file_path.unlink()
                        cleaned_files.append(f"{file_path.name} (0 bytes)")
                    except Exception:
                        pass
                        
            if cleaned_files:
                RichLogger.success(f"Cleaned {len(cleaned_files)} cache files (locks, temp files, etc.)")
                
        except Exception as e:
            RichLogger.warning(f"Could not clean cache locks: {str(e)}")
    
    def _initialize_embeddings_safely(self):
        """Initialize embeddings with multiple fallback attempts"""
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                RichLogger.info(f"Initializing embeddings with model: {self._embedding_model}")
                
                # Force single-threaded tokenizer
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                
                # Initialize with minimal parameters first
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self._embedding_model,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True, 'batch_size': 1}
                )
                
                # Test the embeddings
                test_embedding = self.embeddings.embed_query("test")
                if test_embedding and len(test_embedding) > 0:
                    RichLogger.success("Embeddings initialized successfully")
                    return
                else:
                    raise ValueError("Embeddings returned empty result")
                    
            except Exception as e:
                retry_count += 1
                
                if retry_count == 1:
                    RichLogger.warning(f"Initial embedding initialization failed: {str(e)}")
                    RichLogger.info("Attempting cache cleanup and retry...")
                    
                    # More aggressive cleanup
                    self._deep_cache_cleanup()
                    
                    # Force garbage collection
                    gc.collect()
                    time.sleep(2)  # Give system time to release resources
                    
                elif retry_count <= max_retries:
                    RichLogger.warning(f"Retry {retry_count} failed: {str(e)}")
                    time.sleep(1)
                    
        # If all retries failed
        if self.embeddings is None:
            raise RuntimeError(
                f"Failed to initialize embeddings after {max_retries + 1} attempts. "
                "Try deleting ~/.cache/huggingface/ directory and restarting."
            )
    
    def _deep_cache_cleanup(self):
        """Perform deep cache cleanup for the specific model"""
        try:
            cache_dir = Path.home() / ".cache" / "huggingface"
            model_cache_dirs = [
                cache_dir / "hub" / f"models--{self._embedding_model.replace('/', '--')}",
                cache_dir / "transformers" / f"{self._embedding_model.replace('/', '_')}"
            ]
            
            for model_dir in model_cache_dirs:
                if model_dir.exists():
                    # Remove .lock files and incomplete downloads
                    for lock_file in model_dir.rglob("*.lock"):
                        try:
                            lock_file.unlink()
                        except Exception:
                            pass
                            
                    # Remove incomplete download folders
                    for tmp_dir in model_dir.rglob("tmp*"):
                        if tmp_dir.is_dir():
                            try:
                                shutil.rmtree(tmp_dir)
                            except Exception:
                                pass
                                
        except Exception as e:
            RichLogger.warning(f"Deep cache cleanup encountered error: {str(e)}")
    
    def get_embeddings(self):
        """Get the embeddings instance"""
        if self.embeddings is None:
            raise RuntimeError("Embeddings not initialized. Call initialize() first.")
        return self.embeddings
    
    def get_model_name(self):
        """Get the current embedding model name"""
        return self._embedding_model
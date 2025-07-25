"""Cache management utilities for HuggingFace and other caches"""

import os
import time
import shutil
from pathlib import Path
from typing import List, Tuple

from rag_cli.utils.logger import RichLogger


class CacheManager:
    """Manages various cache operations for the application"""
    
    @staticmethod
    def clean_hf_cache_locks(aggressive: bool = True) -> int:
        """
        Clean HuggingFace cache lock files
        
        Args:
            aggressive: If True, removes locks older than 5 minutes. Otherwise 30 minutes.
            
        Returns:
            Number of files cleaned
        """
        try:
            cache_dir = Path.home() / ".cache" / "huggingface"
            if not cache_dir.exists():
                return 0
                
            cleaned_files = []
            lock_age_threshold = 300 if aggressive else 1800  # 5 or 30 minutes
            
            # Clean lock files and temp files
            for pattern in ["*.lock", "*.tmp*", "tmp_*", "*~"]:
                for lock_file in cache_dir.rglob(pattern):
                    try:
                        # Remove lock files older than threshold
                        if lock_file.stat().st_mtime < (time.time() - lock_age_threshold):
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
                RichLogger.success(f"Cleaned {len(cleaned_files)} cache files")
                
            return len(cleaned_files)
                
        except Exception as e:
            RichLogger.warning(f"Could not clean cache locks: {str(e)}")
            return 0
    
    @staticmethod
    def ensure_cache_directories() -> List[Path]:
        """
        Ensure cache directories exist with proper permissions
        
        Returns:
            List of created/verified cache directories
        """
        cache_dirs = []
        
        # HuggingFace cache
        hf_cache = Path.home() / ".cache" / "huggingface"
        
        # Sentence transformers cache
        st_cache = Path.home() / ".cache" / "torch" / "sentence_transformers"
        
        for cache_dir in [hf_cache, st_cache]:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                # Ensure the directory is writable
                test_file = cache_dir / ".write_test"
                test_file.touch()
                test_file.unlink()
                cache_dirs.append(cache_dir)
            except Exception as e:
                RichLogger.warning(f"Cache directory issue: {cache_dir} - {str(e)}")
                
        return cache_dirs
    
    @staticmethod
    def get_cache_info() -> dict:
        """Get information about cache directories"""
        info = {}
        
        cache_paths = {
            "huggingface": Path.home() / ".cache" / "huggingface",
            "torch": Path.home() / ".cache" / "torch",
            "pip": Path.home() / ".cache" / "pip"
        }
        
        for name, path in cache_paths.items():
            if path.exists():
                try:
                    size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                    info[name] = {
                        "path": str(path),
                        "exists": True,
                        "size_mb": round(size / (1024 * 1024), 2),
                        "file_count": sum(1 for _ in path.rglob('*') if _.is_file())
                    }
                except Exception as e:
                    info[name] = {
                        "path": str(path),
                        "exists": True,
                        "error": str(e)
                    }
            else:
                info[name] = {
                    "path": str(path),
                    "exists": False
                }
                
        return info
    
    @staticmethod
    def clear_model_cache(model_name: str) -> Tuple[bool, str]:
        """
        Clear cache for a specific model
        
        Args:
            model_name: Name of the model to clear cache for
            
        Returns:
            Tuple of (success, message)
        """
        try:
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_dir_name = f"models--{model_name.replace('/', '--')}"
            model_path = cache_dir / model_dir_name
            
            if model_path.exists():
                shutil.rmtree(model_path)
                return True, f"Successfully cleared cache for {model_name}"
            else:
                return False, f"No cache found for {model_name}"
                
        except Exception as e:
            return False, f"Error clearing cache: {str(e)}"
    
    @staticmethod
    def fix_permissions(cache_dir: Path = None) -> int:
        """
        Fix permissions on cache directories
        
        Args:
            cache_dir: Specific cache directory to fix, or None for all
            
        Returns:
            Number of files/directories fixed
        """
        fixed_count = 0
        
        if cache_dir is None:
            cache_dir = Path.home() / ".cache"
            
        try:
            for item in cache_dir.rglob('*'):
                try:
                    # Ensure owner has read/write permissions
                    current_mode = item.stat().st_mode
                    new_mode = current_mode | 0o600  # Add read/write for owner
                    if item.is_dir():
                        new_mode |= 0o100  # Add execute for directories
                    
                    if current_mode != new_mode:
                        os.chmod(item, new_mode)
                        fixed_count += 1
                except Exception:
                    pass
                    
            if fixed_count > 0:
                RichLogger.success(f"Fixed permissions on {fixed_count} items")
                
        except Exception as e:
            RichLogger.warning(f"Error fixing permissions: {str(e)}")
            
        return fixed_count
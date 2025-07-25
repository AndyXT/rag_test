"""Constants and environment setup for RAG CLI Application"""

import os

# Optional clipboard support
try:
    import pyperclip  # For clipboard support
except ImportError:
    pyperclip = None  # Fallback to file export

# Constants
PYPERCLIP_AVAILABLE = pyperclip is not None

# Set environment variables for better cache management and to prevent conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer parallelism issues
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Reduce verbosity
os.environ["HF_DATASETS_OFFLINE"] = "0"  # Allow online access
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # Allow online access
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Disable symlink warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"  # Disable advisory warnings

# Additional settings to help with file descriptor issues
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # Reduce file creation
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Better compatibility
os.environ["OMP_NUM_THREADS"] = "1"  # Reduce threading issues
os.environ["MKL_NUM_THREADS"] = "1"  # Reduce threading issues

# Set cache directories with better control
# Use HF_HOME as the main cache directory (TRANSFORMERS_CACHE is deprecated)
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_DATASETS_CACHE"] = os.path.expanduser("~/.cache/huggingface/datasets")

# Ensure cache directories exist
cache_dirs = [
    os.environ["HF_HOME"],
    os.environ["HF_DATASETS_CACHE"],
    os.path.join(os.environ["HF_HOME"], "hub"),
    os.path.join(os.environ["HF_HOME"], "transformers"),
]
for cache_dir in cache_dirs:
    os.makedirs(cache_dir, exist_ok=True)

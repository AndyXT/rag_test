#!/usr/bin/env python3
"""
Hugging Face Cache Manager
A utility to manage Hugging Face cache without deleting everything.
"""

import os
import time
from pathlib import Path
import argparse
from typing import List


def clean_lock_files(cache_dir: Path) -> List[str]:
    """Clean stale lock files from cache directory"""
    if not cache_dir.exists():
        return []
    
    lock_patterns = ['*.lock', '*.tmp*', '*incomplete*', '*.part']
    cleaned_files = []
    
    try:
        for pattern in lock_patterns:
            for lock_file in cache_dir.rglob(pattern):
                try:
                    if lock_file.is_file():
                        # Check if lock file is stale (older than 30 minutes)
                        if time.time() - lock_file.stat().st_mtime > 1800:
                            print(f"Removing stale lock file: {lock_file}")
                            lock_file.unlink()
                            cleaned_files.append(str(lock_file))
                except (OSError, PermissionError) as e:
                    print(f"Could not remove {lock_file}: {e}")
                    continue
        
        return cleaned_files
        
    except Exception as e:
        print(f"Error cleaning lock files: {e}")
        return []


def validate_cache_structure(cache_dir: Path) -> bool:
    """Validate that cache directory has proper structure"""
    required_dirs = ['hub', 'transformers']
    
    for dir_name in required_dirs:
        dir_path = cache_dir / dir_name
        if not dir_path.exists():
            print(f"Creating missing directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Test if directory is writable
    try:
        test_file = cache_dir / '.write_test'
        test_file.touch()
        test_file.unlink()
        return True
    except (OSError, PermissionError):
        print(f"Cache directory is not writable: {cache_dir}")
        return False


def fix_permissions(cache_dir: Path):
    """Fix permissions on cache directory"""
    try:
        # Set appropriate permissions (user read/write/execute)
        cache_dir.chmod(0o755)
        for item in cache_dir.rglob('*'):
            if item.is_dir():
                item.chmod(0o755)
            else:
                item.chmod(0o644)
        print(f"Fixed permissions for {cache_dir}")
    except (OSError, PermissionError) as e:
        print(f"Could not fix permissions: {e}")


def main():
    parser = argparse.ArgumentParser(description='Manage Hugging Face cache')
    parser.add_argument('--clean-locks', action='store_true', 
                       help='Clean stale lock files')
    parser.add_argument('--validate', action='store_true',
                       help='Validate cache structure')
    parser.add_argument('--fix-permissions', action='store_true',
                       help='Fix cache directory permissions')
    parser.add_argument('--cache-dir', type=str,
                       default=os.path.expanduser('~/.cache/huggingface'),
                       help='Cache directory path')
    
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    print(f"Working with cache directory: {cache_dir}")
    
    if args.clean_locks or not any([args.validate, args.fix_permissions]):
        print("Cleaning stale lock files...")
        cleaned = clean_lock_files(cache_dir)
        if cleaned:
            print(f"Cleaned {len(cleaned)} stale lock files")
        else:
            print("No stale lock files found")
    
    if args.validate:
        print("Validating cache structure...")
        if validate_cache_structure(cache_dir):
            print("Cache structure is valid")
        else:
            print("Cache structure has issues")
    
    if args.fix_permissions:
        print("Fixing permissions...")
        fix_permissions(cache_dir)


if __name__ == '__main__':
    main()
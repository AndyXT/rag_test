#!/usr/bin/env python3
"""HuggingFace Cache Manager - Helps manage and clean the HF cache"""

import os
import shutil
import time
from pathlib import Path
import argparse

def get_cache_dir():
    """Get the HuggingFace cache directory"""
    return Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))

def get_cache_size(path):
    """Calculate total size of a directory"""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except:
        pass
    return total

def format_size(bytes):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

def clean_locks(cache_dir, age_minutes=5):
    """Clean lock and temporary files"""
    patterns = ["*.lock", "*.tmp*", "*incomplete*", "*.part", "*downloading*"]
    cleaned = []
    
    for pattern in patterns:
        for file in cache_dir.rglob(pattern):
            try:
                if file.is_file():
                    # Check age or if zero-byte
                    if (time.time() - file.stat().st_mtime > age_minutes * 60) or file.stat().st_size == 0:
                        file.unlink()
                        cleaned.append(str(file))
            except:
                continue
    
    return cleaned

def validate_cache(cache_dir):
    """Validate cache integrity"""
    issues = []
    
    # Check for common issues
    model_dir = cache_dir / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2"
    if model_dir.exists():
        # Check for lock files
        locks = list(model_dir.rglob("*.lock"))
        if locks:
            issues.append(f"Found {len(locks)} lock files in embeddings model cache")
        
        # Check for incomplete downloads
        incomplete = list(model_dir.rglob("*incomplete*")) + list(model_dir.rglob("*.tmp*"))
        if incomplete:
            issues.append(f"Found {len(incomplete)} incomplete downloads")
    
    return issues

def main():
    parser = argparse.ArgumentParser(description="Manage HuggingFace cache")
    parser.add_argument("--clean-locks", action="store_true", help="Clean lock and temp files")
    parser.add_argument("--validate", action="store_true", help="Validate cache integrity")
    parser.add_argument("--clear-model", help="Clear specific model cache", metavar="MODEL")
    parser.add_argument("--clear-all", action="store_true", help="Clear entire cache (WARNING: will redownload models)")
    parser.add_argument("--info", action="store_true", help="Show cache information")
    parser.add_argument("--fix-permissions", action="store_true", help="Fix cache directory permissions")
    
    args = parser.parse_args()
    
    cache_dir = get_cache_dir()
    
    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        return
    
    # Default action is info if nothing specified
    if not any(vars(args).values()):
        args.info = True
    
    if args.info:
        print(f"🗂️  HuggingFace Cache Information")
        print(f"📍 Location: {cache_dir}")
        size = get_cache_size(cache_dir)
        print(f"💾 Total size: {format_size(size)}")
        
        # Count models
        models = list((cache_dir / "hub").glob("models--*")) if (cache_dir / "hub").exists() else []
        print(f"📦 Cached models: {len(models)}")
        
        # Check for issues
        issues = validate_cache(cache_dir)
        if issues:
            print("\n⚠️  Issues found:")
            for issue in issues:
                print(f"  - {issue}")
    
    if args.validate:
        print("\n🔍 Validating cache...")
        issues = validate_cache(cache_dir)
        if issues:
            print("❌ Issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ Cache appears healthy")
    
    if args.clean_locks:
        print("\n🧹 Cleaning lock and temporary files...")
        cleaned = clean_locks(cache_dir)
        if cleaned:
            print(f"✅ Cleaned {len(cleaned)} files")
            for f in cleaned[:5]:  # Show first 5
                print(f"  - {Path(f).name}")
            if len(cleaned) > 5:
                print(f"  ... and {len(cleaned) - 5} more")
        else:
            print("✅ No files to clean")
    
    if args.clear_model:
        model_path = cache_dir / "hub" / f"models--{args.clear_model.replace('/', '--')}"
        if model_path.exists():
            print(f"\n🗑️  Clearing model cache: {args.clear_model}")
            shutil.rmtree(model_path)
            print("✅ Model cache cleared")
        else:
            print(f"❌ Model not found in cache: {args.clear_model}")
    
    if args.clear_all:
        response = input("\n⚠️  This will delete ALL cached models. Continue? (yes/no): ")
        if response.lower() == "yes":
            print("🗑️  Clearing entire cache...")
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            print("✅ Cache cleared")
        else:
            print("❌ Cancelled")
    
    if args.fix_permissions:
        print("\n🔧 Fixing cache permissions...")
        try:
            # Make cache directories writable
            for path in cache_dir.rglob("*"):
                if path.is_dir():
                    path.chmod(0o755)
                elif path.is_file():
                    path.chmod(0o644)
            print("✅ Permissions fixed")
        except Exception as e:
            print(f"❌ Error fixing permissions: {e}")

if __name__ == "__main__":
    main()
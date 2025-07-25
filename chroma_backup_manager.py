#!/usr/bin/env python3
"""
ChromaDB Backup Manager
A utility to manage ChromaDB backups and restore functionality.
"""

import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple


def list_backups(db_path: str) -> List[Tuple[str, Path]]:
    """List all available backups for a database"""
    base_path = Path(db_path)
    parent_dir = base_path.parent
    backup_pattern = f"{base_path.name}_backup_*"
    
    backups = []
    for item in parent_dir.glob(backup_pattern):
        if item.is_dir():
            try:
                timestamp = int(item.name.split('_backup_')[1])
                backup_time = datetime.fromtimestamp(timestamp)
                backups.append((backup_time.strftime("%Y-%m-%d %H:%M:%S"), item))
            except (ValueError, IndexError):
                continue
    
    # Sort by timestamp (newest first)
    backups.sort(reverse=True)
    return backups


def restore_backup(backup_path: Path, db_path: str) -> bool:
    """Restore a backup to the main database location"""
    try:
        final_path = Path(db_path)
        
        # Create backup of current database if it exists
        if final_path.exists():
            current_backup = Path(str(final_path) + f"_before_restore_{int(datetime.now().timestamp())}")
            shutil.move(str(final_path), str(current_backup))
            print(f"Current database backed up to: {current_backup}")
        
        # Restore the backup
        shutil.copytree(str(backup_path), str(final_path))
        print(f"Successfully restored backup from {backup_path}")
        return True
        
    except Exception as e:
        print(f"Failed to restore backup: {e}")
        return False


def create_manual_backup(db_path: str) -> bool:
    """Create a manual backup of the current database"""
    try:
        source_path = Path(db_path)
        if not source_path.exists():
            print(f"Database does not exist at {db_path}")
            return False
        
        backup_path = Path(str(source_path) + f"_manual_backup_{int(datetime.now().timestamp())}")
        shutil.copytree(str(source_path), str(backup_path))
        print(f"Manual backup created at: {backup_path}")
        return True
        
    except Exception as e:
        print(f"Failed to create backup: {e}")
        return False


def cleanup_backups(db_path: str, max_backups: int = 5) -> int:
    """Clean up old backups, keeping only the most recent ones"""
    base_path = Path(db_path)
    parent_dir = base_path.parent
    backup_pattern = f"{base_path.name}_backup_*"
    
    # Find all backup directories
    backup_dirs = []
    for item in parent_dir.glob(backup_pattern):
        if item.is_dir():
            try:
                timestamp = int(item.name.split('_backup_')[1])
                backup_dirs.append((timestamp, item))
            except (ValueError, IndexError):
                continue
    
    # Sort by timestamp (newest first) and remove old backups
    backup_dirs.sort(reverse=True)
    removed_count = 0
    
    for _, backup_dir in backup_dirs[max_backups:]:
        try:
            shutil.rmtree(str(backup_dir))
            print(f"Removed old backup: {backup_dir.name}")
            removed_count += 1
        except Exception as e:
            print(f"Could not remove backup {backup_dir}: {e}")
    
    return removed_count


def validate_database(db_path: str) -> bool:
    """Validate that a ChromaDB database is properly structured"""
    try:
        db_path = Path(db_path)
        if not db_path.exists():
            print(f"Database path does not exist: {db_path}")
            return False
        
        # Check for essential ChromaDB files
        essential_files = ["chroma.sqlite3"]
        missing_files = []
        
        for file_name in essential_files:
            file_path = db_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"Database is missing essential files: {', '.join(missing_files)}")
            return False
        
        print("Database structure appears valid")
        return True
        
    except Exception as e:
        print(f"Error validating database: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Manage ChromaDB backups')
    parser.add_argument('--db-path', type=str, default='./chroma_db',
                       help='Path to ChromaDB database')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List backups
    subparsers.add_parser('list', help='List available backups')
    
    # Restore backup
    restore_parser = subparsers.add_parser('restore', help='Restore a backup')
    restore_parser.add_argument('backup_name', help='Name of backup to restore')
    
    # Create backup
    subparsers.add_parser('backup', help='Create a manual backup')
    
    # Cleanup backups
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old backups')
    cleanup_parser.add_argument('--keep', type=int, default=5,
                               help='Number of backups to keep (default: 5)')
    
    # Validate database
    subparsers.add_parser('validate', help='Validate database structure')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    db_path = args.db_path
    print(f"Working with database: {db_path}")
    
    if args.command == 'list':
        backups = list_backups(db_path)
        if backups:
            print("\nAvailable backups:")
            for backup_time, backup_path in backups:
                print(f"  {backup_time} - {backup_path.name}")
        else:
            print("No backups found")
    
    elif args.command == 'restore':
        backups = list_backups(db_path)
        backup_to_restore = None
        
        for _, backup_path in backups:
            if args.backup_name in backup_path.name:
                backup_to_restore = backup_path
                break
        
        if backup_to_restore:
            restore_backup(backup_to_restore, db_path)
        else:
            print(f"Backup '{args.backup_name}' not found")
    
    elif args.command == 'backup':
        create_manual_backup(db_path)
    
    elif args.command == 'cleanup':
        removed = cleanup_backups(db_path, args.keep)
        print(f"Removed {removed} old backups")
    
    elif args.command == 'validate':
        validate_database(db_path)


if __name__ == '__main__':
    main()
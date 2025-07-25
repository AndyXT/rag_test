"""Document management service for handling document operations."""

from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

from rag_cli.utils.logger import RichLogger


class DocumentService:
    """Service for document-related operations and validations."""
    
    @staticmethod
    def validate_documents_directory(docs_path: str) -> Tuple[bool, str, List[Path]]:
        """
        Validate documents directory and return status information.
        
        Args:
            docs_path: Path to the documents directory
            
        Returns:
            Tuple of (is_valid, message, pdf_files)
        """
        path = Path(docs_path)
        
        # Check if directory exists
        if not path.exists():
            return False, f"Documents directory not found: {docs_path}", []
        
        if not path.is_dir():
            return False, f"Path is not a directory: {docs_path}", []
        
        # Find PDF files
        pdf_files = list(path.glob("*.pdf"))
        
        if not pdf_files:
            return False, f"No PDF files found in {docs_path}", []
        
        # Success
        message = f"Found {len(pdf_files)} PDF file{'s' if len(pdf_files) > 1 else ''}"
        RichLogger.info(message)
        
        return True, message, pdf_files
    
    @staticmethod
    def get_document_stats(docs_path: str) -> Dict[str, Any]:
        """
        Get statistics about documents in a directory.
        
        Args:
            docs_path: Path to the documents directory
            
        Returns:
            Dictionary with document statistics
        """
        path = Path(docs_path)
        stats = {
            "total_files": 0,
            "total_size_mb": 0,
            "files": []
        }
        
        if not path.exists() or not path.is_dir():
            return stats
        
        pdf_files = list(path.glob("*.pdf"))
        stats["total_files"] = len(pdf_files)
        
        for pdf_file in pdf_files:
            size_mb = pdf_file.stat().st_size / (1024 * 1024)
            stats["total_size_mb"] += size_mb
            stats["files"].append({
                "name": pdf_file.name,
                "size_mb": round(size_mb, 2)
            })
        
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        
        return stats
    
    @staticmethod
    def validate_database_path(db_path: str) -> Tuple[bool, str]:
        """
        Validate database path for creation or loading.
        
        Args:
            db_path: Path to the database directory
            
        Returns:
            Tuple of (is_valid, message)
        """
        path = Path(db_path)
        
        # For loading, check if it exists and contains data
        if path.exists():
            if not path.is_dir():
                return False, f"Database path exists but is not a directory: {db_path}"
            
            # Check for ChromaDB files
            chroma_files = list(path.glob("chroma*"))
            if not chroma_files:
                return False, f"No ChromaDB files found in {db_path}"
            
            return True, f"Found existing database at {db_path}"
        
        # For creation, check if parent directory is writable
        parent = path.parent
        if not parent.exists():
            return False, f"Parent directory does not exist: {parent}"
        
        if not parent.is_dir():
            return False, f"Parent path is not a directory: {parent}"
        
        # Try to create the directory
        try:
            path.mkdir(exist_ok=True)
            return True, f"Database directory ready at {db_path}"
        except Exception as e:
            return False, f"Cannot create database directory: {str(e)}"
    
    @staticmethod
    def estimate_processing_time(pdf_files: List[Path]) -> Tuple[int, str]:
        """
        Estimate processing time for PDF files.
        
        Args:
            pdf_files: List of PDF file paths
            
        Returns:
            Tuple of (estimated_seconds, human_readable_time)
        """
        if not pdf_files:
            return 0, "0 seconds"
        
        # Estimate based on file sizes and count
        total_size_mb = sum(f.stat().st_size / (1024 * 1024) for f in pdf_files)
        
        # Rough estimates: 5 seconds per file + 2 seconds per MB
        estimated_seconds = len(pdf_files) * 5 + int(total_size_mb * 2)
        
        # Convert to human readable
        if estimated_seconds < 60:
            time_str = f"{estimated_seconds} seconds"
        elif estimated_seconds < 3600:
            minutes = estimated_seconds // 60
            seconds = estimated_seconds % 60
            time_str = f"{minutes} minute{'s' if minutes > 1 else ''}"
            if seconds > 0:
                time_str += f" {seconds} seconds"
        else:
            hours = estimated_seconds // 3600
            minutes = (estimated_seconds % 3600) // 60
            time_str = f"{hours} hour{'s' if hours > 1 else ''}"
            if minutes > 0:
                time_str += f" {minutes} minute{'s' if minutes > 1 else ''}"
        
        return estimated_seconds, time_str
    
    @staticmethod
    def check_disk_space(path: str, required_mb: float = 500) -> Tuple[bool, str]:
        """
        Check if there's enough disk space for database creation.
        
        Args:
            path: Path where database will be created
            required_mb: Required space in MB (default 500MB)
            
        Returns:
            Tuple of (has_space, message)
        """
        import shutil
        
        try:
            path_obj = Path(path)
            # Get the mount point for the path
            if path_obj.exists():
                check_path = path_obj
            else:
                check_path = path_obj.parent
                
            stat = shutil.disk_usage(check_path)
            available_mb = stat.free / (1024 * 1024)
            
            if available_mb < required_mb:
                return False, f"Insufficient disk space. Need {required_mb}MB, have {available_mb:.0f}MB"
            
            return True, f"Sufficient disk space available ({available_mb:.0f}MB)"
            
        except Exception as e:
            RichLogger.warning(f"Could not check disk space: {str(e)}")
            return True, "Could not verify disk space (proceeding anyway)"
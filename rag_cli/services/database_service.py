"""Service for handling database operations"""

from typing import Optional, Dict, Any, Callable
from pathlib import Path

from rag_cli.utils.logger import RichLogger
from rag_cli.utils.defaults import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


class DatabaseService:
    """Handles database creation, loading, and management"""
    
    def __init__(self, vectorstore_manager: Any, settings_manager: Any):
        self.vectorstore_manager = vectorstore_manager
        self.settings_manager = settings_manager
    
    def initialize_vectorstore(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> None:
        """
        Initialize the vectorstore manager with settings
        
        Args:
            chunk_size: Override chunk size
            chunk_overlap: Override chunk overlap
        """
        # Get settings with defaults
        chunk_size = chunk_size or self.settings_manager.get("chunk_size", DEFAULT_CHUNK_SIZE)
        chunk_overlap = chunk_overlap or self.settings_manager.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
        
        # Initialize vectorstore manager
        self.vectorstore_manager.initialize(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        RichLogger.info(f"Vectorstore initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def load_database(self, db_path: str = "./chroma_db") -> bool:
        """
        Load existing database
        
        Args:
            db_path: Path to the database
            
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            RichLogger.info(f"Loading database from {db_path}...")
            
            # Check if database exists
            if not Path(db_path).exists():
                RichLogger.warning(f"Database not found at {db_path}")
                return False
            
            # Load database
            success = self.vectorstore_manager.load_existing_db(db_path)
            
            if success:
                stats = self.get_database_stats()
                if stats:
                    RichLogger.success(
                        f"Database loaded successfully with {stats.get('document_count', 'unknown')} documents"
                    )
                else:
                    RichLogger.success("Database loaded successfully")
            else:
                RichLogger.error("Failed to load database")
            
            return success
            
        except Exception as e:
            RichLogger.error(f"Error loading database: {str(e)}")
            return False
    
    def create_database(
        self,
        docs_path: str = "./documents",
        db_path: str = "./chroma_db",
        progress_callback: Optional[Callable] = None
    ) -> bool:
        """
        Create new database from documents
        
        Args:
            docs_path: Path to documents directory
            db_path: Path where database will be created
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if successfully created, False otherwise
        """
        try:
            # Validate documents directory
            docs_path_obj = Path(docs_path)
            if not docs_path_obj.exists():
                error_msg = f"Documents directory not found: {docs_path}"
                RichLogger.error(error_msg)
                if progress_callback:
                    progress_callback(f"Error: {error_msg}")
                return False
            
            # Check for PDF files
            pdf_count = len(list(docs_path_obj.glob("**/*.pdf")))
            if pdf_count == 0:
                error_msg = f"No PDF files found in {docs_path}"
                RichLogger.error(error_msg)
                if progress_callback:
                    progress_callback(f"Error: {error_msg}")
                return False
            
            RichLogger.info(f"Creating database from {pdf_count} PDF files...")
            if progress_callback:
                progress_callback(f"Starting database creation with {pdf_count} PDF files...")
            
            # Create database
            self.vectorstore_manager.create_db_from_docs(
                docs_path=docs_path,
                db_path=db_path,
                progress_callback=progress_callback
            )
            
            RichLogger.success("Database created successfully")
            return True
            
        except Exception as e:
            error_msg = f"Error creating database: {str(e)}"
            RichLogger.error(error_msg)
            if progress_callback:
                progress_callback(f"Error: {error_msg}")
            return False
    
    def get_database_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get database statistics
        
        Returns:
            Dictionary with database stats or None
        """
        try:
            stats = self.vectorstore_manager.get_stats()
            return stats
        except Exception as e:
            RichLogger.warning(f"Could not get database stats: {str(e)}")
            return None
    
    def validate_database(self) -> bool:
        """
        Validate that database is properly loaded and functional
        
        Returns:
            True if database is valid and functional
        """
        try:
            # Check if vectorstore exists
            if not self.vectorstore_manager.get_vectorstore():
                return False
            
            # Try to get stats
            stats = self.get_database_stats()
            if not stats:
                return False
            
            # Check document count
            doc_count = stats.get("document_count", 0)
            if isinstance(doc_count, str) or doc_count == 0:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get comprehensive database information
        
        Returns:
            Dictionary with database information
        """
        info = {
            "loaded": False,
            "document_count": 0,
            "chunk_size": self.settings_manager.get("chunk_size", DEFAULT_CHUNK_SIZE),
            "chunk_overlap": self.settings_manager.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
            "embedding_model": self.settings_manager.get("embedding_model", "Unknown"),
            "valid": False
        }
        
        try:
            if self.vectorstore_manager.get_vectorstore():
                info["loaded"] = True
                
                stats = self.get_database_stats()
                if stats:
                    info.update(stats)
                
                info["valid"] = self.validate_database()
            
        except Exception as e:
            RichLogger.warning(f"Error getting database info: {str(e)}")
        
        return info
    
    def reset_database(self) -> None:
        """Reset database by clearing the vectorstore"""
        try:
            # Clear vectorstore reference
            if hasattr(self.vectorstore_manager, 'chroma_manager'):
                self.vectorstore_manager.chroma_manager.vectorstore = None
            
            RichLogger.info("Database reset successfully")
            
        except Exception as e:
            RichLogger.error(f"Error resetting database: {str(e)}")
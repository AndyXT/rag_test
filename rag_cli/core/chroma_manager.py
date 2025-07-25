"""ChromaDB database management"""

import os
import gc
import time
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Any

from langchain_chroma import Chroma
from langchain.schema import Document

from rag_cli.utils.logger import RichLogger
from rag_cli.utils.defaults import DEFAULT_BATCH_SIZE


class ChromaManager:
    """Manages ChromaDB database operations"""
    
    def __init__(self):
        self.vectorstore = None
        self._setup_environment()
    
    def _setup_environment(self) -> None:
        """Set up environment variables for ChromaDB"""
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        os.environ["CHROMA_SERVER_NOFILE"] = "65536"  # Increase file descriptor limit
    
    def load_existing_db(self, db_path: str, embeddings: Any) -> bool:
        """
        Load existing ChromaDB
        
        Args:
            db_path: Path to the database
            embeddings: Embeddings instance to use
            
        Returns:
            True if successfully loaded, False otherwise
        """
        if not os.path.exists(db_path):
            return False
            
        try:
            # Force garbage collection before loading
            gc.collect()
            
            self.vectorstore = Chroma(
                persist_directory=db_path, 
                embedding_function=embeddings
            )
            
            # Force garbage collection after loading
            gc.collect()
            
            RichLogger.success(f"Successfully loaded database from {db_path}")
            return True
            
        except Exception as e:
            RichLogger.warning(f"Could not load database: {str(e)}")
            if "fds_to_keep" in str(e):
                RichLogger.warning("💡 Try restarting the application or increasing file descriptor limit")
            return False
    
    def create_database(
        self, 
        texts: List[Document], 
        embeddings: Any,
        db_path: str,
        batch_size: int = DEFAULT_BATCH_SIZE
    ) -> Chroma:
        """
        Create ChromaDB with backup and validation
        
        Args:
            texts: Documents to add to the database
            embeddings: Embeddings instance
            db_path: Final database path
            batch_size: Batch size for processing
            
        Returns:
            Created Chroma vectorstore
        """
        final_path = Path(db_path)
        backup_path = None
        temp_path = None
        
        try:
            # Clean up old backups first
            self._cleanup_old_backups(db_path)
            
            # Create backup of existing database
            backup_path = self._create_backup_if_exists(db_path)
            
            # Create database in temporary location
            self.vectorstore, temp_path = self._create_temp_database(
                texts, embeddings, batch_size
            )
            
            # Validate the temporary database
            if not self._validate_temp_database(temp_path):
                raise Exception("Created database appears to be incomplete or corrupted")
            
            # Move to final location and clean up
            self._finalize_database(temp_path, final_path, backup_path)
            
            # Final garbage collection
            gc.collect()
            return self.vectorstore
            
        except Exception as e:
            # Clean up temporary directory on failure
            if temp_path and temp_path.exists():
                try:
                    shutil.rmtree(str(temp_path))
                    RichLogger.info("Cleaned up temporary database")
                except Exception:
                    pass
            
            # Restore backup if it exists
            self._restore_backup_on_failure(backup_path, final_path)
            
            gc.collect()
            raise Exception(f"ChromaDB creation failed: {str(e)}") from e
    
    def _create_backup_if_exists(self, db_path: str) -> Optional[Path]:
        """Create backup of existing database if it exists"""
        final_path = Path(db_path)
        if not final_path.exists():
            return None
            
        backup_path = Path(str(final_path) + f"_backup_{int(time.time())}")
        RichLogger.info(f"Backing up existing database to {backup_path}")
        shutil.move(str(final_path), str(backup_path))
        return backup_path
    
    def _create_temp_database(
        self, 
        texts: List[Document], 
        embeddings: Any,
        batch_size: int
    ) -> Tuple[Chroma, Path]:
        """Create database in temporary directory with batch processing"""
        temp_dir = tempfile.mkdtemp(prefix="chroma_temp_")
        temp_path = Path(temp_dir)
        
        vectorstore = None
        total_batches = (len(texts) + batch_size - 1) // batch_size
        delay_between_batches = 0.05
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            
            vectorstore = self._process_batch(
                vectorstore, batch, embeddings, str(temp_path), 
                batch_size
            )
            
            # Force garbage collection and small delay between batches
            gc.collect()
            if batch_num < total_batches:
                time.sleep(delay_between_batches)
        
        if vectorstore is None:
            raise Exception(
                "Failed to create any vectorstore - all batches failed. "
                "This may be due to embedding model issues or document processing problems."
            )
        
        return vectorstore, temp_path
    
    def _process_batch(
        self, 
        vectorstore: Optional[Chroma], 
        batch: List[Document],
        embeddings: Any, 
        persist_dir: str, 
        batch_size: int
    ) -> Optional[Chroma]:
        """Process a single batch of documents"""
        try:
            if vectorstore is None:
                # Create initial vectorstore
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=persist_dir,
                    collection_metadata={"hnsw:space": "cosine"},
                )
            else:
                # Add to existing vectorstore
                vectorstore.add_documents(batch)
            return vectorstore
            
        except Exception:
            # If batch fails, try with smaller batches
            if batch_size > 5:
                return self._retry_with_smaller_batches(
                    vectorstore, batch, embeddings, persist_dir, 
                    batch_size
                )
            return vectorstore
    
    def _retry_with_smaller_batches(
        self, 
        vectorstore: Optional[Chroma],
        failed_batch: List[Document], 
        embeddings: Any,
        persist_dir: str, 
        original_batch_size: int
    ) -> Optional[Chroma]:
        """Retry failed batch with smaller size"""
        smaller_batch_size = max(5, original_batch_size // 2)
        
        for j in range(0, len(failed_batch), smaller_batch_size):
            small_batch = failed_batch[j : j + smaller_batch_size]
            try:
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        documents=small_batch,
                        embedding=embeddings,
                        persist_directory=persist_dir,
                        collection_metadata={"hnsw:space": "cosine"},
                    )
                else:
                    vectorstore.add_documents(small_batch)
                gc.collect()
                time.sleep(0.05)
            except Exception:
                # Skip this small batch and continue
                continue
        
        return vectorstore
    
    def _validate_temp_database(self, temp_path: Path) -> bool:
        """Validate that the temporary database is complete"""
        if not temp_path or not temp_path.exists():
            return False
            
        essential_files = ["chroma.sqlite3"]
        return all((temp_path / f).exists() for f in essential_files)
    
    def _finalize_database(
        self, 
        temp_path: Path, 
        final_path: Path,
        backup_path: Optional[Path]
    ) -> None:
        """Move temp database to final location and clean up backup"""
        shutil.move(str(temp_path), str(final_path))
        RichLogger.success(f"Database successfully created at {final_path}")
        
        # Remove backup if creation was successful
        if backup_path and backup_path.exists():
            try:
                shutil.rmtree(str(backup_path))
                RichLogger.success("Removed backup database")
            except Exception:
                RichLogger.warning(f"Could not remove backup at {backup_path}")
    
    def _restore_backup_on_failure(
        self, 
        backup_path: Optional[Path],
        final_path: Path
    ) -> None:
        """Restore backup database on failure"""
        if backup_path and backup_path.exists():
            try:
                shutil.move(str(backup_path), str(final_path))
                RichLogger.success("Restored backup database")
            except Exception as restore_error:
                RichLogger.error(f"Failed to restore backup: {restore_error}")
    
    def _cleanup_old_backups(self, db_path: str, max_backups: int = 3) -> None:
        """Clean up old backup directories to prevent disk space issues"""
        try:
            base_path = Path(db_path)
            parent_dir = base_path.parent
            backup_pattern = f"{base_path.name}_backup_*"

            # Find all backup directories
            backup_dirs = []
            for item in parent_dir.glob(backup_pattern):
                if item.is_dir():
                    try:
                        # Extract timestamp from backup name
                        timestamp = int(item.name.split("_backup_")[1])
                        backup_dirs.append((timestamp, item))
                    except (ValueError, IndexError):
                        continue

            # Sort by timestamp (newest first) and remove old backups
            backup_dirs.sort(reverse=True)
            for _, backup_dir in backup_dirs[max_backups:]:
                try:
                    shutil.rmtree(str(backup_dir))
                    RichLogger.info(f"Removed old backup: {backup_dir.name}")
                except Exception as e:
                    RichLogger.warning(f"Could not remove old backup {backup_dir}: {e}")

        except Exception as e:
            RichLogger.warning(f"Error during backup cleanup: {e}")
    
    def get_vectorstore(self) -> Optional[Chroma]:
        """Get the current vectorstore instance"""
        return self.vectorstore
    
    def get_stats(self, chunk_size: int = None, chunk_overlap: int = None) -> Optional[dict]:
        """Get database statistics"""
        if not self.vectorstore:
            return None

        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            stats = {"document_count": count}
            
            if chunk_size is not None:
                stats["chunk_size"] = chunk_size
            if chunk_overlap is not None:
                stats["chunk_overlap"] = chunk_overlap
                
            return stats
        except Exception:
            return {"document_count": "Unknown"}
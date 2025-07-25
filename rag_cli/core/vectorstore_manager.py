"""Vector store management for RAG system"""

import os
import gc
import resource
from typing import Optional, List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from rag_cli.utils.logger import RichLogger
from rag_cli.utils.defaults import (
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
)

# Import focused managers
from rag_cli.core.embeddings_manager import EmbeddingsManager
from rag_cli.core.cache_manager import CacheManager
from rag_cli.core.pdf_processor import PDFProcessor
from rag_cli.core.chroma_manager import ChromaManager
from rag_cli.core.reranker_manager import RerankerManager


class VectorStoreManager:
    """Manages vector store operations and document processing"""

    def __init__(self, settings_manager: Optional[Any] = None) -> None:
        self.settings_manager = settings_manager
        self.chunk_size = DEFAULT_CHUNK_SIZE
        self.chunk_overlap = DEFAULT_CHUNK_OVERLAP
        
        # Initialize embeddings and chroma managers only
        self.embeddings_manager = EmbeddingsManager(settings_manager)
        self.chroma_manager = ChromaManager()
        self.reranker_manager = RerankerManager(settings_manager)
        
        # Document processing will be handled separately
        from rag_cli.core.document_processor import DocumentProcessor
        self.document_processor = DocumentProcessor(
            PDFProcessor(),
            self.embeddings_manager,
            CacheManager(),
            self.chunk_size,
            self.chunk_overlap
        )

    def initialize(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> None:
        """Initialize or update settings"""
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
            
        # Update document processor settings
        self.document_processor.update_chunk_settings(self.chunk_size, self.chunk_overlap)
        
        # Check and increase file descriptor limit
        self._check_and_increase_fd_limit()
        
        # Initialize embeddings
        self.embeddings_manager.initialize()

    def _check_and_increase_fd_limit(self) -> None:
        """Check and try to increase file descriptor limit"""
        try:
            # Get current limits
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

            # Try to increase to a reasonable limit
            target_limit = min(8192, hard)

            if soft < target_limit:
                try:
                    resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, hard))
                    RichLogger.success(f"Increased file descriptor limit from {soft} to {target_limit}")
                except Exception:
                    RichLogger.warning(f"Could not increase file descriptor limit (current: {soft})")
                    RichLogger.warning("💡 Try running: ulimit -n 8192 before starting the app")
        except Exception:
            # Not critical if this fails
            pass

    def load_existing_db(self, db_path: str = "./chroma_db") -> bool:
        """Load existing ChromaDB with modern configuration"""
        # Clean HF cache before loading to prevent embedding issues
        RichLogger.info("Cleaning HuggingFace cache before loading database...")
        cache_manager = CacheManager()
        cache_manager.clean_hf_cache_locks(aggressive=True)

        # Set environment variables for ChromaDB (modern approach)
        os.environ["ANONYMIZED_TELEMETRY"] = "False"

        # Get embeddings from the embeddings manager
        embeddings = self.embeddings_manager.get_embeddings()
        
        # Delegate to chroma manager
        result = self.chroma_manager.load_existing_db(db_path, embeddings)
        
        return result

    def create_db_from_docs(
        self, docs_path: str = "./documents", db_path: str = "./chroma_db", progress_callback=None
    ):
        """Create new ChromaDB from documents with robust error handling"""
        try:
            self._setup_environment()
            
            if progress_callback:
                progress_callback("Processing documents...")
            
            # Use document processor to handle all document operations
            all_documents = self.document_processor.process_documents(docs_path, progress_callback)
            
            if progress_callback:
                progress_callback(f"Creating embeddings for {len(all_documents)} chunks...")
            
            # Get embeddings
            embeddings = self.embeddings_manager.get_embeddings()
            
            # Create database using chroma manager
            self.chroma_manager.create_database(all_documents, embeddings, db_path)
            
            # Report success
            if progress_callback:
                progress_callback(f"Successfully created database with {len(all_documents)} chunks")

        except Exception as e:
            # Force cleanup on error
            gc.collect()
            # Re-raise with more context
            raise Exception(f"Database creation failed: {str(e)}") from e

    def _setup_environment(self):
        """Set up environment variables for document processing"""
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"

    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using the cross-encoder model"""
        # Delegate to reranker manager
        return self.reranker_manager.rerank_documents(query, documents)

    def get_vectorstore(self):
        """Get the current vectorstore instance"""
        return self.chroma_manager.get_vectorstore()

    def get_embeddings(self):
        """Get the embeddings instance"""
        return self.embeddings_manager.get_embeddings()

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get database statistics"""
        return self.chroma_manager.get_stats(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
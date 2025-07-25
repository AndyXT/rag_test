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
        
        # Initialize focused managers
        self.embeddings_manager = EmbeddingsManager(settings_manager)
        self.cache_manager = CacheManager()
        self.pdf_processor = PDFProcessor()
        self.chroma_manager = ChromaManager()
        self.reranker_manager = RerankerManager(settings_manager)

    def initialize(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:
        """Initialize the vector store manager"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Check and increase file descriptor limit before initialization
        self._check_and_increase_fd_limit()
        
        # Initialize embeddings using the focused manager
        self.embeddings_manager.initialize()
        
        # Initialize reranker if enabled
        self.reranker_manager.initialize()

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
        self.cache_manager.clean_hf_cache_locks(aggressive=True)

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
        """Create new ChromaDB from documents with robust error handling and file descriptor management"""
        try:
            self._setup_environment()
            
            # Validate and get PDF files
            _, pdf_files = self.pdf_processor.validate_pdf_directory(docs_path)
            
            if progress_callback:
                progress_callback(f"Found {len(pdf_files)} PDF files...")
            
            # Process PDFs using the PDF processor
            all_documents = self.pdf_processor.process_pdfs(pdf_files, progress_callback)
            
            # Get processing summary
            summary = self.pdf_processor.get_processing_summary()
            
            if progress_callback:
                msg = f"Successfully loaded {summary['success_count']} files"
                if summary['failure_count'] > 0:
                    msg += f" ({summary['failure_count']} failed)"
                progress_callback(msg)
            
            # Split documents
            texts = self._split_documents(all_documents, progress_callback)
            
            if progress_callback:
                progress_callback(f"Creating embeddings for {len(texts)} chunks...")
            
            # Get embeddings
            embeddings = self.embeddings_manager.get_embeddings()
            
            # Create database using chroma manager
            self.chroma_manager.create_database(texts, embeddings, db_path)
            
            # Report success
            self._report_success(texts, pdf_files, progress_callback)

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

    def _split_documents(self, all_documents, progress_callback):
        """Split documents into chunks"""
        if progress_callback:
            progress_callback(f"Splitting {len(all_documents)} documents...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        texts = text_splitter.split_documents(all_documents)

        # Clear large document list to free memory
        del all_documents
        gc.collect()

        return texts

    def _report_success(self, texts, _, progress_callback):
        """Report successful database creation"""
        summary = self.pdf_processor.get_processing_summary()
        
        success_msg = f"Database created with {len(texts)} chunks from {summary['success_count']} files"
        if summary['failure_count'] > 0:
            success_msg += f" ({summary['failure_count']} files skipped due to errors)"

        if progress_callback:
            progress_callback(success_msg)

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
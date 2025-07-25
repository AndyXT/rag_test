"""Document processing logic separated from VectorStoreManager."""

from pathlib import Path
from typing import List, Optional, Callable
import gc

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_cli.utils.logger import RichLogger
from rag_cli.core.pdf_processor import PDFProcessor
from rag_cli.core.embeddings_manager import EmbeddingsManager
from rag_cli.core.cache_manager import CacheManager


class DocumentProcessor:
    """Handles document loading, splitting, and embedding generation."""
    
    def __init__(
        self, 
        pdf_processor: PDFProcessor,
        embeddings_manager: EmbeddingsManager,
        cache_manager: CacheManager,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize DocumentProcessor.
        
        Args:
            pdf_processor: PDF processing instance
            embeddings_manager: Embeddings management instance
            cache_manager: Cache management instance
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.pdf_processor = pdf_processor
        self.embeddings_manager = embeddings_manager
        self.cache_manager = cache_manager
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def process_documents(
        self, 
        docs_path: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> List[Document]:
        """
        Process all documents in a directory.
        
        Args:
            docs_path: Path to documents directory
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of processed documents
        """
        # Validate and get PDF files
        docs_path_obj, pdf_files = self.pdf_processor.validate_pdf_directory(docs_path)
        
        if progress_callback:
            progress_callback(f"Found {len(pdf_files)} PDF files to process")
        
        # Process PDFs in batches
        all_documents = []
        batch_size = 5  # Process 5 PDFs at a time
        
        for i in range(0, len(pdf_files), batch_size):
            batch = pdf_files[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(pdf_files) + batch_size - 1) // batch_size
            
            if progress_callback:
                progress_callback(f"Processing batch {batch_num}/{total_batches}...")
            
            # Load documents from batch
            batch_documents = self._load_batch_documents(batch, progress_callback)
            
            # Split documents
            if progress_callback:
                progress_callback(f"Splitting documents in batch {batch_num}...")
            
            split_docs = self._split_documents(batch_documents)
            all_documents.extend(split_docs)
            
            # Clean up after each batch
            self._cleanup_batch()
            
            RichLogger.info(f"Processed batch {batch_num}/{total_batches}: {len(split_docs)} chunks")
        
        if progress_callback:
            progress_callback(f"Processed {len(all_documents)} document chunks total")
        
        return all_documents
    
    def _load_batch_documents(
        self, 
        pdf_files: List[Path],
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> List[Document]:
        """
        Load documents from a batch of PDF files.
        
        Args:
            pdf_files: List of PDF file paths
            progress_callback: Optional progress callback
            
        Returns:
            List of loaded documents
        """
        documents = []
        
        for pdf_file in pdf_files:
            try:
                if progress_callback:
                    progress_callback(f"Loading: {pdf_file.name}")
                
                docs = self.pdf_processor.load_pdf_incremental(str(pdf_file))
                documents.extend(docs)
                
            except Exception as e:
                RichLogger.error(f"Failed to load {pdf_file.name}: {str(e)}")
                continue
        
        return documents
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split documents
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        return text_splitter.split_documents(documents)
    
    def _cleanup_batch(self):
        """Clean up resources after processing a batch."""
        try:
            # Clean cache locks
            self.cache_manager.clean_hf_cache_locks()
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            RichLogger.warning(f"Cleanup warning: {str(e)}")
    
    def update_chunk_settings(self, chunk_size: int, chunk_overlap: int):
        """
        Update chunk size and overlap settings.
        
        Args:
            chunk_size: New chunk size
            chunk_overlap: New chunk overlap
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
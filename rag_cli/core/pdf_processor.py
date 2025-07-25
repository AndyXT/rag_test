"""PDF processing utilities for document loading"""

import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

from langchain_community.document_loaders import PyPDFLoader




class PDFProcessor:
    """Handles PDF processing and document extraction"""
    
    def __init__(self):
        self.successful_files: List[str] = []
        self.failed_files: List[str] = []
    
    @contextmanager
    def _pdf_loader_context(self, pdf_path: Path):
        """Context manager for PDF loader to ensure cleanup"""
        loader = PyPDFLoader(str(pdf_path))
        try:
            yield loader
        finally:
            # Cleanup any resources
            del loader
            gc.collect()
    
    def process_pdfs(
        self, 
        pdf_files: List[Path], 
        progress_callback: Optional[Any] = None
    ) -> List[Any]:
        """
        Process multiple PDF files and extract documents
        
        Args:
            pdf_files: List of PDF file paths
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of extracted documents
        """
        all_documents = []
        self.successful_files = []
        self.failed_files = []
        
        for i, pdf_file in enumerate(pdf_files):
            self._process_single_pdf(
                pdf_file, 
                i, 
                len(pdf_files), 
                all_documents, 
                progress_callback
            )
        
        if not all_documents:
            self._handle_no_documents_error()
        
        return all_documents
    
    def _process_single_pdf(
        self, 
        pdf_file: Path, 
        index: int, 
        total: int, 
        all_documents: List[Any], 
        progress_callback: Optional[Any]
    ) -> None:
        """Process a single PDF file"""
        try:
            if progress_callback:
                progress_callback(
                    f"Processing {pdf_file.name} ({index+1}/{total})..."
                )

            # Use context manager for proper cleanup
            with self._pdf_loader_context(pdf_file) as loader:
                documents = loader.load()

                if documents:
                    all_documents.extend(documents)
                    self.successful_files.append(pdf_file.name)
                else:
                    self.failed_files.append(f"{pdf_file.name} (no content)")

            # Force garbage collection after each PDF to free file descriptors
            gc.collect()

        except Exception as pdf_error:
            self.failed_files.append(f"{pdf_file.name} ({str(pdf_error)[:50]}...)")
            gc.collect()  # Clean up even on error
    
    def _handle_no_documents_error(self) -> None:
        """Handle case where no documents could be processed"""
        error_msg = "No documents could be processed successfully."
        if self.failed_files:
            error_msg += f" Failed files: {', '.join(self.failed_files[:3])}"
            if len(self.failed_files) > 3:
                error_msg += f" and {len(self.failed_files) - 3} more"
        raise ValueError(error_msg)
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of PDF processing results"""
        return {
            "successful_files": self.successful_files,
            "failed_files": self.failed_files,
            "success_count": len(self.successful_files),
            "failure_count": len(self.failed_files),
            "total_processed": len(self.successful_files) + len(self.failed_files)
        }
    
    @staticmethod
    def validate_pdf_directory(docs_path: str) -> Tuple[Path, List[Path]]:
        """
        Validate PDF directory and return list of PDF files
        
        Args:
            docs_path: Path to documents directory
            
        Returns:
            Tuple of (directory Path, list of PDF file Paths)
            
        Raises:
            ValueError: If directory doesn't exist or contains no PDFs
        """
        docs_path = Path(docs_path)
        
        if not docs_path.exists():
            raise ValueError(f"Documents directory '{docs_path}' not found.")
            
        if not docs_path.is_dir():
            raise ValueError(f"'{docs_path}' is not a directory.")
        
        pdf_files = list(docs_path.glob("**/*.pdf"))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {docs_path}")
            
        return docs_path, pdf_files
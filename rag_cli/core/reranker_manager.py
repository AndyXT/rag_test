"""Reranker model management for document reranking"""

import asyncio
from typing import List, Optional, Any

try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False

from langchain.schema import Document

from rag_cli.utils.logger import RichLogger
from rag_cli.utils.defaults import DEFAULT_RERANKER_MODEL


class RerankerManager:
    """Manages document reranking operations"""
    
    def __init__(self, settings_manager: Optional[Any] = None):
        self.settings_manager = settings_manager
        self.reranker = None
        self._reranker_model = None
        
    def initialize(self) -> bool:
        """
        Initialize reranker model if enabled
        
        Returns:
            True if successfully initialized, False otherwise
        """
        if not self.settings_manager:
            return False
            
        use_reranker = self.settings_manager.get("use_reranker", False)
        if not use_reranker:
            return False
            
        if not CROSSENCODER_AVAILABLE:
            RichLogger.warning("CrossEncoder not available. Install sentence-transformers to use reranking.")
            return False
            
        self._reranker_model = self.settings_manager.get("reranker_model", DEFAULT_RERANKER_MODEL)
        
        try:
            RichLogger.info(f"Initializing reranker model: {self._reranker_model}")
            self.reranker = CrossEncoder(
                self._reranker_model,
                max_length=512,
                device="cpu",  # Force CPU to avoid GPU memory issues
                trust_remote_code=True  # Required for Qwen models
            )
            RichLogger.success("Reranker initialized successfully")
            return True
            
        except Exception as e:
            RichLogger.error(f"Failed to initialize reranker: {str(e)}")
            RichLogger.warning("Continuing without reranking")
            self.reranker = None
            return False
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Rerank documents using the cross-encoder model
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return (defaults to settings)
            
        Returns:
            Reranked list of documents
        """
        if not self.reranker or not documents:
            return documents
            
        try:
            # Get reranker settings
            if top_k is None:
                top_k = self.settings_manager.get("reranker_top_k", 3) if self.settings_manager else 3
            
            # Prepare query-document pairs
            pairs = [[query, doc.page_content] for doc in documents]
            
            # Get reranking scores with timeout
            RichLogger.info(f"Reranking {len(documents)} documents...")
            
            scores = self._predict_with_timeout(pairs, timeout=15.0)
            
            if scores is None:
                # Timeout occurred, return top k without reranking
                RichLogger.error("Reranking timed out after 15 seconds")
                return documents[:top_k]
            
            # Sort documents by score (higher is better for cross-encoder)
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top k documents
            reranked_docs = [doc for doc, score in doc_scores[:top_k]]
            
            # Log reranking results
            self._log_reranking_results(doc_scores, top_k)
            
            return reranked_docs
            
        except Exception as e:
            RichLogger.error(f"Reranking failed: {str(e)}")
            RichLogger.warning("Using original document order")
            return documents[:top_k] if top_k else documents
    
    def _predict_with_timeout(self, pairs: List[List[str]], timeout: float) -> Optional[List[float]]:
        """
        Run prediction with timeout
        
        Args:
            pairs: Query-document pairs
            timeout: Timeout in seconds
            
        Returns:
            List of scores or None if timeout
        """
        try:
            # Create new event loop for prediction
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run prediction in executor with timeout
            future = loop.run_in_executor(None, self.reranker.predict, pairs)
            scores = loop.run_until_complete(asyncio.wait_for(future, timeout=timeout))
            
            return scores
            
        except asyncio.TimeoutError:
            return None
            
        finally:
            loop.close()
    
    def _log_reranking_results(self, doc_scores: List[tuple], top_k: int) -> None:
        """Log reranking results for debugging"""
        RichLogger.success(f"Reranked to top {min(len(doc_scores), top_k)} documents")
        
        for i, (doc, score) in enumerate(doc_scores[:top_k]):
            preview = doc.page_content[:80].replace('\n', ' ')
            RichLogger.debug(f"Rank {i+1}: score={score:.3f} - {preview}...")
    
    def is_available(self) -> bool:
        """Check if reranker is available and initialized"""
        return self.reranker is not None
    
    def get_model_name(self) -> Optional[str]:
        """Get the current reranker model name"""
        return self._reranker_model
    
    @staticmethod
    def is_library_available() -> bool:
        """Check if the CrossEncoder library is available"""
        return CROSSENCODER_AVAILABLE
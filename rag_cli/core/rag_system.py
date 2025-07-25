# RAG System Core Module
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import our logger and defaults
from rag_cli.utils.logger import RichLogger
from rag_cli.utils.defaults import (
    DEFAULT_OLLAMA_MODEL, DEFAULT_TEMPERATURE, DEFAULT_CHUNK_SIZE, 
    DEFAULT_CHUNK_OVERLAP, DEFAULT_RETRIEVAL_K
)

# LangChain imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# Import our managers
from .llm_manager import LLMManager
from .vectorstore_manager import VectorStoreManager

# Import processors
from .query_processor import QueryProcessor
from .error_handler import ErrorHandler


class RAGSystem:
    """Enhanced RAG System with modern configuration and robust error handling"""

    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        retrieval_k: int = DEFAULT_RETRIEVAL_K,
        settings_manager: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        self.settings_manager = settings_manager
        self.qa_chain = None
        self.conversation_history = []
        
        # Initialize managers
        self.llm_manager = LLMManager(settings_manager)
        self.vectorstore_manager = VectorStoreManager(settings_manager)
        
        # Initialize processors
        self.query_processor = QueryProcessor(settings_manager)
        self.error_handler = ErrorHandler(model_name)
        
        # Initialize new focused components
        from rag_cli.core.query_executor import QueryExecutor
        from rag_cli.core.retrieval_manager import RetrievalManager
        
        self.query_executor = QueryExecutor(
            self.llm_manager, 
            self.query_processor,
            self.error_handler
        )
        self.retrieval_manager = RetrievalManager(
            self.vectorstore_manager,
            self.query_processor
        )
        
        # Initialize the managers
        self.llm_manager.initialize(model_name, temperature)
        self.vectorstore_manager.initialize(chunk_size, chunk_overlap)

    def update_settings(self, **kwargs) -> None:
        """Update RAG system settings"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Reinitialize managers with new settings
        self.llm_manager.initialize(self.model_name, self.temperature)
        self.vectorstore_manager.initialize(self.chunk_size, self.chunk_overlap)

        # Recreate QA chain if vectorstore exists
        if self.vectorstore:
            self._setup_qa_chain()

    @property
    def llm(self):
        """Get the LLM instance from the manager"""
        return self.llm_manager.get_llm()
    
    @property
    def embeddings(self):
        """Get the embeddings instance from the manager"""
        return self.vectorstore_manager.get_embeddings()
    
    @property
    def vectorstore(self):
        """Get the vectorstore instance from the manager"""
        return self.vectorstore_manager.get_vectorstore()
    
    @property
    def reranker(self):
        """Get the reranker instance from the manager"""
        return self.vectorstore_manager.reranker
    
    @property
    def query_expansion_llm(self):
        """Get the query expansion LLM from the manager"""
        return self.llm_manager.get_query_expansion_llm()

    def load_existing_db(self, db_path: str = "./chroma_db") -> bool:
        """Load existing ChromaDB with modern configuration"""
        result = self.vectorstore_manager.load_existing_db(db_path)
        if result:
            self._setup_qa_chain()
        return result

    def create_db_from_docs(
        self, docs_path: str = "./documents", db_path: str = "./chroma_db", 
        progress_callback: Optional[Any] = None
    ) -> None:
        """Create new ChromaDB from documents with robust error handling and file descriptor management"""
        self.vectorstore_manager.create_db_from_docs(docs_path, db_path, progress_callback)
        if progress_callback:
            progress_callback("Setting up QA chain...")
        self._setup_qa_chain()

    def _expand_query(self, original_query: str) -> List[str]:
        """Expand a query using the small LLM to improve retrieval"""
        return self.query_processor.expand_query(original_query, self.query_expansion_llm)

    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using the cross-encoder model"""
        return self.vectorstore_manager.rerank_documents(query, documents)

    def _create_expanded_retriever(self, k: int) -> BaseRetriever:
        """Create a retriever that uses query expansion if enabled"""
        return self.retrieval_manager.create_expanded_retriever(k, self.query_expansion_llm)
    
    def _load_system_prompt(self, prompt_file: str = "system_prompt.md") -> str:
        """Load system prompt from file or return default.
        
        Args:
            prompt_file: Path to the system prompt file
            
        Returns:
            System prompt string with {context} placeholder
        """
        from rag_cli.utils.logger import RichLogger
        
        default_prompt = (
            "You are a helpful assistant. Use the following pieces of retrieved context to answer the question. "
            "Provide a comprehensive answer that fully addresses the question. "
            "If you don't know the answer based on the context, say so.\n\n"
            "{context}"
        )
        
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            return default_prompt
            
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Ensure {context} placeholder is present
                if "{context}" not in content:
                    content += "\n\nContext:\n{context}"
                RichLogger.success(f"Loaded custom system prompt from {prompt_file}")
                return content
        except Exception as e:
            RichLogger.warning(f"Failed to load {prompt_file}: {str(e)}")
            return default_prompt
    
    def _setup_qa_chain(self) -> None:
        """Setup the QA chain using modern LangChain approach"""
        RichLogger.info(f"Setting up QA chain with retrieval_k={self.retrieval_k}")
        
        # Load system prompt from file if available
        system_prompt = self._load_system_prompt()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)

        # Use expanded retriever if query expansion is enabled
        retriever = self._create_expanded_retriever(self.retrieval_k)
        
        self.qa_chain = create_retrieval_chain(
            retriever,
            question_answer_chain,
        )

    def _should_use_rag(self) -> bool:
        """Check if RAG should be used."""
        return self.vectorstore is not None and hasattr(self.qa_chain, 'invoke')
    
    async def _execute_qa_chain(self, question: str) -> Dict[str, Any]:
        """Execute query using the QA chain."""
        return await self.query_executor.execute_qa_chain(
            self.qa_chain, 
            question, 
            self._rerank_documents
        )
    
    def _log_relevance_scores(self, question: str) -> None:
        """Log document relevance scores for debugging."""
        self.retrieval_manager.log_relevance_scores(question, self.retrieval_k)
    
    async def _execute_manual_retrieval(self, question: str) -> Dict[str, Any]:
        """Execute manual retrieval when QA chain is not available."""
        retriever = self._create_expanded_retriever(self.retrieval_k)
        system_prompt = self._load_system_prompt()
        
        return await self.query_executor.execute_manual_retrieval(
            retriever,
            question,
            self._rerank_documents,
            system_prompt
        )
    
    def _handle_file_descriptor_error(self, question: str) -> Dict[str, Any]:
        """Handle file descriptor errors with fallback."""
        return self.query_executor.handle_file_descriptor_error(question)
    
    def _get_connection_error_message(self, provider: str) -> str:
        """Get provider-specific connection error messages."""
        from rag_cli.utils.error_utils import ErrorUtils
        
        error_info = ErrorUtils.handle_connection_error(provider)
        return f"{error_info['message']}. {error_info['suggestion']}"
    
    def _execute_query_in_thread(self, question: str) -> Dict[str, Any]:
        """Execute query in a thread to avoid file descriptor issues."""
        import gc
        import os
        
        # Set conservative environment for the query thread
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        try:
            # Force garbage collection
            gc.collect()
            
            # Use QA chain if available, otherwise fall back to manual retrieval
            if self.qa_chain:
                result = self._execute_qa_chain_sync(question)
            else:
                result = self._execute_manual_retrieval_sync(question)
            
            # Clean up
            gc.collect()
            
            return result
            
        except Exception as e:
            error_str = str(e)
            RichLogger.warning(f"Query exception: {error_str[:200]}")
            
            if "fds_to_keep" in error_str:
                return self._handle_file_descriptor_error(question)
            else:
                raise e
    
    def _execute_qa_chain_sync(self, question: str) -> Dict[str, Any]:
        """Synchronous version of _execute_qa_chain for thread execution."""
        return self.query_executor.execute_qa_chain_sync(
            self.qa_chain,
            question,
            self._rerank_documents
        )
    
    def _execute_manual_retrieval_sync(self, question: str) -> Dict[str, Any]:
        """Synchronous version of _execute_manual_retrieval for thread execution."""
        retriever = self._create_expanded_retriever(self.retrieval_k)
        system_prompt = self._load_system_prompt()
        
        return self.query_executor.execute_manual_retrieval_sync(
            retriever,
            question,
            self._rerank_documents,
            system_prompt
        )
    
    async def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with better error handling and async execution"""
        if not self.vectorstore:
            return {"response": "RAG system not initialized. Load or create a database first.", "context": []}

        try:
            # Run the query in a thread executor to avoid blocking the UI
            loop = asyncio.get_event_loop()
            
            # Execute query in thread pool
            result = await loop.run_in_executor(None, self._execute_query_in_thread, question)
            return result

        except Exception as e:
            return self._handle_query_error(e)
    
    def _handle_query_error(self, error: Exception) -> Dict[str, Any]:
        """Handle query errors with appropriate messages."""
        # Update error handler with current model
        self.error_handler.model_name = self.model_name
        
        # Get structured error information
        error_info = self.error_handler.handle_error(error)
        
        # Format error for user
        user_message = self.error_handler.format_error_for_user(error_info)
        
        return {
            "response": user_message,
            "context": [],
            "error_info": error_info
        }

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get database statistics"""
        stats = self.vectorstore_manager.get_stats()
        if stats:
            # Add model and temperature info
            stats["model"] = self.llm_manager.get_current_model_name()
            stats["temperature"] = self.temperature
            stats["retrieval_k"] = self.retrieval_k
        return stats
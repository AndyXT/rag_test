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


class ExpandedRetriever(BaseRetriever):
    """Custom retriever that expands queries using LLM"""
    
    def __init__(self, rag_system, base_retriever, k, **kwargs):
        super().__init__(**kwargs)
        self.rag_system = rag_system
        self.base_retriever = base_retriever
        self.k = k
        
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Expand the query
        expanded_queries = self.rag_system._expand_query(query)
        
        # Retrieve documents for each query
        all_docs = []
        doc_contents_seen = set()
        max_docs_per_query = 10  # Limit docs per query
        max_total_docs = 30  # Hard limit on total docs
        
        for expanded_query in expanded_queries:
            if len(all_docs) >= max_total_docs:
                RichLogger.warning(f"Reached maximum document limit ({max_total_docs}), stopping retrieval")
                break
                
            try:
                docs = self.base_retriever.get_relevant_documents(expanded_query)
                added_count = 0
                for doc in docs[:max_docs_per_query]:  # Limit per query
                    # Deduplicate by content
                    content_hash = hash(doc.page_content)
                    if content_hash not in doc_contents_seen:
                        doc_contents_seen.add(content_hash)
                        all_docs.append(doc)
                        added_count += 1
                        if len(all_docs) >= max_total_docs:
                            break
                RichLogger.info(f"Query '{expanded_query[:50]}...' added {added_count} new documents")
            except Exception as e:
                RichLogger.warning(f"Failed to retrieve for query: {expanded_query[:50]}... - {str(e)}")
        
        RichLogger.info(f"Query expansion retrieved {len(all_docs)} unique documents from {len(expanded_queries)} queries")
        
        # Return requested number of documents
        return all_docs[:self.k]
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        # For async, just call sync version
        return self._get_relevant_documents(query)


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

    def _expand_query(self, original_query):
        """Expand a query using the small LLM to improve retrieval"""
        return self.query_processor.expand_query(original_query, self.query_expansion_llm)

    def _rerank_documents(self, query, documents):
        """Rerank documents using the cross-encoder model"""
        return self.vectorstore_manager.rerank_documents(query, documents)

    def _create_expanded_retriever(self, k):
        """Create a retriever that uses query expansion if enabled"""
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        if not self.query_expansion_llm:
            return base_retriever
            
        return ExpandedRetriever(self, base_retriever, k)
    
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
    
    def _setup_qa_chain(self):
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
        result = self.qa_chain.invoke({"input": question})
        
        # Extract answer and source documents
        answer_text = result.get("answer", "No answer generated")
        source_docs = result.get("context", [])
        
        # Process result using shared logic
        answer_text, source_docs = self._process_qa_result(answer_text, source_docs, question)
        
        return {"response": answer_text, "context": source_docs}
    
    def _log_relevance_scores(self, question: str) -> None:
        """Log document relevance scores for debugging."""
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=self.retrieval_k)
            # Convert to documents with metadata
            documents = []
            for doc, score in docs_with_scores:
                doc.metadata = doc.metadata or {}
                doc.metadata['score'] = score
                documents.append(doc)
            # Use query processor to log scores
            self.query_processor.log_relevance_scores(documents[:3])
        except Exception as e:
            RichLogger.warning(f"Could not get relevance scores: {str(e)}")
    
    async def _execute_manual_retrieval(self, question: str) -> Dict[str, Any]:
        """Execute manual retrieval when QA chain is not available."""
        # Use expanded retriever if query expansion is enabled
        retriever = self._create_expanded_retriever(self.retrieval_k)
        
        # Get relevant documents using shared retrieval logic
        relevant_docs = self._perform_retrieval(retriever, question)
        
        if not relevant_docs:
            return {
                "response": "I couldn't find any relevant information in the documents to answer your question.", 
                "context": []
            }
        
        # Apply reranking if enabled
        if self.reranker and relevant_docs:
            relevant_docs = self._rerank_documents(question, relevant_docs)
        
        # Format context and create prompt
        context = self._format_context(relevant_docs)
        prompt = self._create_prompt(context, question)
        
        # Query LLM
        response = self.llm_manager.invoke(prompt)
        
        return {"response": response, "context": relevant_docs}
    
    def _format_context(self, documents: List) -> str:
        """Format documents into context string."""
        return self.query_processor.format_context(documents)
    
    def _create_prompt(self, context: str, question: str) -> str:
        """Create prompt with context and question."""
        system_prompt = self._load_system_prompt()
        return self.query_processor.create_rag_prompt(question, context, system_prompt)

    def _perform_retrieval(self, retriever, question):
        """Shared retrieval logic with fallback"""
        try:
            return retriever.invoke(question)
        except Exception as e:
            from rag_cli.utils.logger import RichLogger
            RichLogger.warning(f"Retriever invoke failed, using fallback: {str(e)}")
            return retriever.get_relevant_documents(question)

    def _process_qa_result(self, answer_text, source_docs, question):
        """Process QA chain result with reranking and conversion"""
        # Apply reranking if enabled
        if self.reranker and source_docs:
            source_docs = self._rerank_documents(question, source_docs)
        
        # Log relevance scores for debugging
        self._log_relevance_scores(question)
        
        # Convert response to string if it's an object
        if hasattr(answer_text, 'content'):
            answer_text = answer_text.content
        else:
            answer_text = str(answer_text)
        
        return answer_text, source_docs
    
    def _handle_file_descriptor_error(self, question: str) -> Dict[str, Any]:
        """Handle file descriptor errors with fallback."""
        RichLogger.warning("Falling back to simple query without retrieval due to file descriptor error")
        
        # Create a specific file descriptor error
        fd_error = Exception("fds_to_keep error - file descriptor limit exceeded")
        error_info = self.error_handler.handle_error(fd_error)
        
        # Try fallback query
        try:
            simple_prompt = f"Question: {question}\n\nPlease provide a helpful response based on general knowledge."
            response = self.llm_manager.invoke(simple_prompt)
            return {"response": response, "context": [], "error_info": error_info}
        except Exception:
            # Return formatted error message
            return {
                "response": self.error_handler.format_error_for_user(error_info), 
                "context": [],
                "error_info": error_info
            }
    
    def _get_connection_error_message(self, provider: str) -> str:
        """Get provider-specific connection error messages."""
        messages = {
            "ollama": (
                "Cannot connect to Ollama. You can:\n"
                "1. Start Ollama (run 'ollama serve' in terminal)\n"
                "2. Install the model (run 'ollama pull llama3.2')\n"
                "3. Or switch to an API provider in Settings (Ctrl+S)"
            ),
            "openai": (
                "Cannot connect to OpenAI API. Please check:\n"
                "1. Your API key is correct in Settings (Ctrl+S)\n"
                "2. Your internet connection is working\n"
                "3. The API endpoint URL is correct (if using custom endpoint)\n"
                "4. Or switch to Ollama in Settings (Ctrl+S)"
            ),
            "anthropic": (
                "Cannot connect to Anthropic API. Please check:\n"
                "1. Your API key is correct in Settings (Ctrl+S)\n"
                "2. Your internet connection is working\n"
                "3. Or switch to Ollama in Settings (Ctrl+S)"
            )
        }
        return messages.get(provider, (
            "Connection error. Please check your network connection and try again.\n"
            "You can also switch providers in Settings (Ctrl+S)"
        ))
    
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
        result = self.qa_chain.invoke({"input": question})
        
        # Extract answer and source documents
        answer_text = result.get("answer", "No answer generated")
        source_docs = result.get("context", [])
        
        # Process result using shared logic
        answer_text, source_docs = self._process_qa_result(answer_text, source_docs, question)
        
        RichLogger.info(f"QA chain retrieved {len(source_docs)} documents (retrieval_k={self.retrieval_k})")
        
        return {"response": answer_text, "context": source_docs}
    
    def _execute_manual_retrieval_sync(self, question: str) -> Dict[str, Any]:
        """Synchronous version of _execute_manual_retrieval for thread execution."""
        # Use expanded retriever if query expansion is enabled
        retriever = self._create_expanded_retriever(self.retrieval_k)
        
        # Get relevant documents using shared retrieval logic
        relevant_docs = self._perform_retrieval(retriever, question)
        
        if not relevant_docs:
            return {
                "response": "I couldn't find any relevant information in the documents to answer your question.", 
                "context": []
            }
        
        # Apply reranking if enabled
        if self.reranker and relevant_docs:
            relevant_docs = self._rerank_documents(question, relevant_docs)
        
        # Format context and create prompt
        context = self._format_context(relevant_docs)
        prompt = self._create_prompt(context, question)
        
        # Query LLM
        response = self.llm_manager.invoke(prompt)
        
        return {"response": response, "context": relevant_docs}
    
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

    def get_stats(self):
        """Get database statistics"""
        stats = self.vectorstore_manager.get_stats()
        if stats:
            # Add model and temperature info
            stats["model"] = self.llm_manager.get_current_model_name()
            stats["temperature"] = self.temperature
            stats["retrieval_k"] = self.retrieval_k
        return stats
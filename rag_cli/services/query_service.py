"""Service for handling query processing and retrieval"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document

from rag_cli.utils.logger import RichLogger
from rag_cli.utils.defaults import DEFAULT_TEMPERATURE, DEFAULT_RETRIEVAL_K
from rag_cli.core.query_processor import QueryProcessor
from rag_cli.core.error_handler import ErrorHandler


class QueryService:
    """Handles query processing, expansion, and retrieval logic"""
    
    def __init__(self, rag_system: Any):
        self.rag_system = rag_system
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.query_processor = QueryProcessor(rag_system.settings_manager)
        self.error_handler = ErrorHandler(rag_system.model_name)
    
    async def process_query(
        self,
        question: str,
        use_rag: bool = True,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process a query and return response with metadata
        
        Args:
            question: The user's question
            use_rag: Whether to use RAG for retrieval
            temperature: Override temperature for this query
            
        Returns:
            Dict containing response and metadata
        """
        try:
            # Override temperature if specified
            original_temp = None
            if temperature is not None:
                original_temp = self.rag_system.temperature
                self.rag_system.temperature = temperature
            
            # Execute query based on RAG usage
            if use_rag and self._should_use_rag():
                result = await self._execute_rag_query(question)
            else:
                result = await self._execute_direct_query(question)
            
            # Restore original temperature
            if original_temp is not None:
                self.rag_system.temperature = original_temp
            
            return result
            
        except Exception as e:
            RichLogger.error(f"Query processing error: {str(e)}")
            return self._handle_query_error(e)
    
    def _should_use_rag(self) -> bool:
        """Check if RAG retrieval should be used"""
        return (
            self.rag_system.vectorstore is not None and
            self.rag_system.qa_chain is not None
        )
    
    async def _execute_rag_query(self, question: str) -> Dict[str, Any]:
        """Execute query with RAG retrieval"""
        loop = asyncio.get_event_loop()
        
        # Run the synchronous chain in executor
        future = loop.run_in_executor(
            self.executor,
            self._execute_rag_sync,
            question
        )
        
        return await future
    
    def _execute_rag_sync(self, question: str) -> Dict[str, Any]:
        """Synchronous RAG execution"""
        try:
            # Use the QA chain if available (which includes ExpandedRetriever)
            if self.rag_system.qa_chain:
                # Use the QA chain which handles retrieval internally
                result = self.rag_system.qa_chain.invoke({"input": question})
                
                # Extract documents from the result
                docs = result.get("context", []) if isinstance(result, dict) else []
                
                # Extract documents from the chain result
                if isinstance(result, dict) and "context" in result:
                    # The retrieval chain returns documents in 'context'
                    docs = result.get("context", [])
                else:
                    docs = []
                
                # Get the answer text
                answer_text = result.get("answer", "") if isinstance(result, dict) else str(result)
                
                # Format the response to match UI expectations
                return {
                    "response": answer_text,  # UI expects 'response' not 'answer'
                    "answer": answer_text,    # Keep for backward compatibility
                    "context": docs,          # UI expects 'context' not 'source_documents'
                    "source_documents": docs, # Keep for backward compatibility
                    "method": "rag",
                    "metadata": {
                        "model": self.rag_system.model_name,
                        "temperature": self.rag_system.temperature,
                        "retrieval_k": len(docs),
                        "reranking_used": self.rag_system.settings_manager.get("use_reranker", False),
                        "query_expansion_used": self.rag_system.settings_manager.get("use_query_expansion", False),
                        "query_refinement_used": self.rag_system.settings_manager.get("use_query_refinement", False)
                    }
                }
            else:
                # Fallback to direct retrieval if no QA chain
                # Expand query if enabled
                expanded_query = self._expand_query_if_enabled(question)
                
                # Get retrieval settings
                retrieval_k = self.rag_system.settings_manager.get("retrieval_k", DEFAULT_RETRIEVAL_K)
                
                # Perform retrieval
                docs = self.rag_system.vectorstore.similarity_search(
                    expanded_query, 
                    k=retrieval_k
                )
                
                # Rerank documents if enabled
                docs = self._rerank_if_enabled(expanded_query, docs)
                
                # Format context
                context = self._format_retrieved_context(docs)
                
                # Create prompt and get response
                prompt = self._create_rag_prompt(question, context)
                response = self.rag_system.llm.invoke(prompt)
                
                # Process and return result
                return self._process_rag_result(response, docs, context)
            
        except Exception as e:
            RichLogger.error(f"RAG execution error: {str(e)}")
            raise
    
    async def _execute_direct_query(self, question: str) -> Dict[str, Any]:
        """Execute direct query without RAG"""
        loop = asyncio.get_event_loop()
        
        future = loop.run_in_executor(
            self.executor,
            self._execute_direct_sync,
            question
        )
        
        return await future
    
    def _execute_direct_sync(self, question: str) -> Dict[str, Any]:
        """Synchronous direct query execution"""
        try:
            response = self.rag_system.llm.invoke(question)
            
            return {
                "response": response,      # UI expects 'response'
                "answer": response,
                "context": [],             # UI expects 'context'
                "source_documents": [],
                "context_text": None,
                "method": "direct",
                "metadata": {
                    "model": self.rag_system.model_name,
                    "temperature": self.rag_system.temperature
                }
            }
        except Exception as e:
            RichLogger.error(f"Direct query error: {str(e)}")
            raise
    
    def _expand_query_if_enabled(self, query: str) -> str:
        """Expand or refine query based on settings"""
        # Check if query refinement is enabled (preferred over expansion)
        if self.rag_system.settings_manager.get("use_query_refinement", False):
            if self.rag_system.query_expansion_llm:  # Same LLM used for refinement
                return self.query_processor.refine_query(query, self.rag_system.query_expansion_llm)
        
        # Otherwise check if query expansion is enabled
        elif self.rag_system.settings_manager.get("use_query_expansion", False):
            if self.rag_system.query_expansion_llm:
                return self.query_processor.expand_query(query, self.rag_system.query_expansion_llm)
            
        return query
    
    def _rerank_if_enabled(
        self, 
        query: str, 
        docs: List[Document]
    ) -> List[Document]:
        """Rerank documents if reranking is enabled"""
        if not self.rag_system.settings_manager.get("use_reranker", False):
            return docs
            
        # Apply reranking
        reranked_docs = self.rag_system._rerank_documents(query, docs)
        
        # Apply additional filtering if configured
        min_score = self.rag_system.settings_manager.get("min_relevance_score", 0.0)
        if min_score > 0:
            reranked_docs = self.query_processor.filter_documents_by_score(reranked_docs, min_score)
        
        # Apply deduplication if enabled
        if self.rag_system.settings_manager.get("deduplicate_docs", True):
            reranked_docs = self.query_processor.deduplicate_documents(reranked_docs)
        
        return reranked_docs
    
    def _format_retrieved_context(self, docs: List[Document]) -> str:
        """Format retrieved documents into context string"""
        return self.query_processor.format_context(docs)
    
    def _create_rag_prompt(self, question: str, context: str) -> str:
        """Create prompt with context for RAG"""
        system_prompt = self.rag_system._load_system_prompt()
        return self.query_processor.create_rag_prompt(question, context, system_prompt)
    
    def _process_rag_result(
        self, 
        response: str, 
        docs: List[Document],
        context: str
    ) -> Dict[str, Any]:
        """Process RAG result into standardized format"""
        # Log relevance scores if available
        self._log_relevance_scores(docs)
        
        return {
            "response": response,      # UI expects 'response'
            "answer": response,        # Keep for backward compatibility
            "context": docs,           # UI expects 'context' with Document objects
            "source_documents": docs,  # Keep for backward compatibility
            "context_text": context,   # Formatted text version of context
            "method": "rag",
            "metadata": {
                "model": self.rag_system.model_name,
                "temperature": self.rag_system.temperature,
                "retrieval_k": len(docs),
                "reranking_used": self.rag_system.settings_manager.get("use_reranker", False),
                "query_expansion_used": self.rag_system.settings_manager.get("use_query_expansion", False)
            }
        }
    
    def _log_relevance_scores(self, docs: List[Document]) -> None:
        """Log relevance scores if available"""
        self.query_processor.log_relevance_scores(docs)
    
    def _handle_query_error(self, error: Exception) -> Dict[str, Any]:
        """Handle query errors and return error response"""
        # Update error handler model name
        self.error_handler.model_name = self.rag_system.model_name
        
        # Get structured error information
        error_info = self.error_handler.handle_error(error)
        
        # Format error for user
        user_message = self.error_handler.format_error_for_user(error_info)
        
        return {
            "response": user_message,      # UI expects 'response'
            "answer": user_message,
            "context": [],                 # UI expects 'context'
            "source_documents": [],
            "context_text": None,
            "method": "error",
            "metadata": {
                "error": str(error),
                "error_type": type(error).__name__,
                "error_info": error_info
            }
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
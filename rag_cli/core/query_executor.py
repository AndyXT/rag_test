"""Query execution logic separated from RAGSystem for better modularity."""

from typing import Dict, Any, List, Optional
from langchain.schema import Document

from rag_cli.utils.logger import RichLogger


class QueryExecutor:
    """Handles query execution strategies (QA chain vs manual retrieval)."""
    
    def __init__(self, llm_manager, query_processor, error_handler):
        """
        Initialize QueryExecutor with required managers.
        
        Args:
            llm_manager: LLM manager instance
            query_processor: Query processor instance
            error_handler: Error handler instance
        """
        self.llm_manager = llm_manager
        self.query_processor = query_processor
        self.error_handler = error_handler
    
    async def execute_qa_chain(self, qa_chain, question: str, reranker=None) -> Dict[str, Any]:
        """
        Execute query using the QA chain.
        
        Args:
            qa_chain: The QA chain to use
            question: User's question
            reranker: Optional reranker for documents
            
        Returns:
            Dict with response and context
        """
        result = qa_chain.invoke({"input": question})
        
        # Extract answer and source documents
        answer_text = result.get("answer", "No answer generated")
        source_docs = result.get("context", [])
        
        # Process result
        answer_text, source_docs = self._process_qa_result(
            answer_text, source_docs, question, reranker
        )
        
        return {"response": answer_text, "context": source_docs}
    
    def execute_qa_chain_sync(self, qa_chain, question: str, reranker=None) -> Dict[str, Any]:
        """
        Synchronous version of execute_qa_chain for thread execution.
        
        Args:
            qa_chain: The QA chain to use
            question: User's question
            reranker: Optional reranker for documents
            
        Returns:
            Dict with response and context
        """
        result = qa_chain.invoke({"input": question})
        
        # Extract answer and source documents
        answer_text = result.get("answer", "No answer generated")
        source_docs = result.get("context", [])
        
        # Process result
        answer_text, source_docs = self._process_qa_result(
            answer_text, source_docs, question, reranker
        )
        
        RichLogger.info(f"QA chain retrieved {len(source_docs)} documents")
        
        return {"response": answer_text, "context": source_docs}
    
    async def execute_manual_retrieval(
        self, retriever, question: str, reranker=None, system_prompt: str = None
    ) -> Dict[str, Any]:
        """
        Execute manual retrieval when QA chain is not available.
        
        Args:
            retriever: Document retriever
            question: User's question
            reranker: Optional reranker for documents
            system_prompt: Optional system prompt
            
        Returns:
            Dict with response and context
        """
        # Get relevant documents
        relevant_docs = self._perform_retrieval(retriever, question)
        
        if not relevant_docs:
            return {
                "response": "I couldn't find any relevant information in the documents to answer your question.", 
                "context": []
            }
        
        # Apply reranking if enabled
        if reranker and relevant_docs:
            relevant_docs = reranker(question, relevant_docs)
        
        # Format context and create prompt
        context = self.query_processor.format_context(relevant_docs)
        prompt = self.query_processor.create_rag_prompt(question, context, system_prompt)
        
        # Query LLM
        response = self.llm_manager.invoke(prompt)
        
        return {"response": response, "context": relevant_docs}
    
    def execute_manual_retrieval_sync(
        self, retriever, question: str, reranker=None, system_prompt: str = None
    ) -> Dict[str, Any]:
        """
        Synchronous version of execute_manual_retrieval for thread execution.
        
        Args:
            retriever: Document retriever
            question: User's question
            reranker: Optional reranker for documents
            system_prompt: Optional system prompt
            
        Returns:
            Dict with response and context
        """
        # Get relevant documents
        relevant_docs = self._perform_retrieval(retriever, question)
        
        if not relevant_docs:
            return {
                "response": "I couldn't find any relevant information in the documents to answer your question.", 
                "context": []
            }
        
        # Apply reranking if enabled
        if reranker and relevant_docs:
            relevant_docs = reranker(question, relevant_docs)
        
        # Format context and create prompt
        context = self.query_processor.format_context(relevant_docs)
        prompt = self.query_processor.create_rag_prompt(question, context, system_prompt)
        
        # Query LLM
        response = self.llm_manager.invoke(prompt)
        
        return {"response": response, "context": relevant_docs}
    
    def _perform_retrieval(self, retriever, question: str) -> List[Document]:
        """
        Perform document retrieval with fallback.
        
        Args:
            retriever: Document retriever
            question: User's question
            
        Returns:
            List of retrieved documents
        """
        try:
            return retriever.invoke(question)
        except Exception as e:
            RichLogger.warning(f"Retriever invoke failed, using fallback: {str(e)}")
            try:
                return retriever.get_relevant_documents(question)
            except Exception as e2:
                RichLogger.error(f"Retrieval failed: {str(e2)}")
                return []
    
    def _process_qa_result(
        self, answer_text: Any, source_docs: List[Document], question: str, reranker=None
    ) -> tuple[str, List[Document]]:
        """
        Process QA chain result with reranking and conversion.
        
        Args:
            answer_text: Raw answer from QA chain
            source_docs: Source documents
            question: Original question
            reranker: Optional reranker function
            
        Returns:
            Tuple of (processed answer text, processed documents)
        """
        # Apply reranking if enabled
        if reranker and source_docs:
            source_docs = reranker(question, source_docs)
        
        # Convert response to string if it's an object
        if hasattr(answer_text, 'content'):
            answer_text = answer_text.content
        else:
            answer_text = str(answer_text)
        
        return answer_text, source_docs
    
    def handle_file_descriptor_error(self, question: str) -> Dict[str, Any]:
        """
        Handle file descriptor errors with fallback.
        
        Args:
            question: User's question
            
        Returns:
            Dict with response and error info
        """
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
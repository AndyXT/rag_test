# RAG System Core Module
import os
import gc
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

# Rich imports for output formatting
from rich import print

# LangChain imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# Import our managers
from .llm_manager import LLMManager
from .vectorstore_manager import VectorStoreManager


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
                print(f"[yellow]⚠ Reached maximum document limit ({max_total_docs}), stopping retrieval[/yellow]")
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
                print(f"[blue]ℹ Query '{expanded_query[:50]}...' added {added_count} new documents[/blue]")
            except Exception as e:
                print(f"[yellow]⚠ Failed to retrieve for query: {expanded_query[:50]}... - {str(e)}[/yellow]")
        
        print(f"[INFO] Query expansion retrieved {len(all_docs)} unique documents from {len(expanded_queries)} queries")
        
        # Return requested number of documents
        return all_docs[:self.k]
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        # For async, just call sync version
        return self._get_relevant_documents(query)


class RAGSystem:
    """Enhanced RAG System with modern configuration and robust error handling"""

    def __init__(
        self,
        model_name="llama3.2:3b",
        temperature=0.1,
        chunk_size=1000,
        chunk_overlap=200,
        retrieval_k=3,
        settings_manager=None,
    ):
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
        
        # Initialize the managers
        self.llm_manager.initialize(model_name, temperature)
        self.vectorstore_manager.initialize(chunk_size, chunk_overlap)

    def update_settings(self, **kwargs):
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

    def load_existing_db(self, db_path="./chroma_db"):
        """Load existing ChromaDB with modern configuration"""
        result = self.vectorstore_manager.load_existing_db(db_path)
        if result:
            self._setup_qa_chain()
        return result

    def create_db_from_docs(
        self, docs_path="./documents", db_path="./chroma_db", progress_callback=None
    ):
        """Create new ChromaDB from documents with robust error handling and file descriptor management"""
        self.vectorstore_manager.create_db_from_docs(docs_path, db_path, progress_callback)
        if progress_callback:
            progress_callback("Setting up QA chain...")
        self._setup_qa_chain()

    def _expand_query(self, original_query):
        """Expand a query using the small LLM to improve retrieval"""
        expansion_count = self.settings_manager.get("expansion_queries", 3) if self.settings_manager else 3
        return self.llm_manager.expand_query(original_query, expansion_count)

    def _rerank_documents(self, query, documents):
        """Rerank documents using the cross-encoder model"""
        return self.vectorstore_manager.rerank_documents(query, documents)

    def _create_expanded_retriever(self, k):
        """Create a retriever that uses query expansion if enabled"""
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        if not self.query_expansion_llm:
            return base_retriever
            
        return ExpandedRetriever(self, base_retriever, k)
    
    def _setup_qa_chain(self):
        """Setup the QA chain using modern LangChain approach"""
        print(f"[INFO] Setting up QA chain with retrieval_k={self.retrieval_k}")
        
        # Load system prompt from file if available
        system_prompt_file = Path("system_prompt.md")
        if system_prompt_file.exists():
            try:
                with open(system_prompt_file, 'r', encoding='utf-8') as f:
                    system_prompt_content = f.read()
                # Extract just the content before {context} placeholder
                if "{context}" in system_prompt_content:
                    system_prompt = system_prompt_content
                else:
                    # Add context placeholder if not present
                    system_prompt = system_prompt_content + "\n\nContext:\n{context}"
                print("[green]✓ Loaded custom system prompt from system_prompt.md[/green]")
            except Exception as e:
                print(f"[yellow]⚠ Failed to load system_prompt.md: {str(e)}[/yellow]")
                # Fallback to default prompt
                system_prompt = (
                    "You are a helpful assistant. Use the following pieces of retrieved context to answer the question. "
                    "Provide a comprehensive answer that fully addresses the question. "
                    "If you don't know the answer based on the context, say so.\n\n"
                    "{context}"
                )
        else:
            # Default prompt without length restrictions
            system_prompt = (
                "You are a helpful assistant. Use the following pieces of retrieved context to answer the question. "
                "Provide a comprehensive answer that fully addresses the question. "
                "If you don't know the answer based on the context, say so.\n\n"
                "{context}"
            )

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

    async def query(self, question):
        """Query the RAG system with better error handling and async execution"""
        if not self.vectorstore:
            return {"response": "RAG system not initialized. Load or create a database first.", "context": []}

        try:
            # Run the query in a thread executor to avoid blocking the UI
            loop = asyncio.get_event_loop()

            # Create a simple query function that avoids file descriptor issues
            def simple_query():
                import gc
                import os

                # Set conservative environment for the query thread
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

                try:
                    # Force garbage collection
                    gc.collect()

                    # Check if we have qa_chain first
                    if self.qa_chain:
                        # Use the qa_chain which should return both answer and source documents
                        result = self.qa_chain.invoke({"input": question})
                        
                        # Extract answer and source documents
                        answer_text = result.get("answer", "No answer generated")
                        
                        # The qa_chain returns context as a list of documents
                        source_docs = result.get("context", [])
                        
                        # Apply reranking if enabled
                        if self.reranker and source_docs:
                            source_docs = self._rerank_documents(question, source_docs)
                        
                        # Try to get relevance scores using similarity search
                        try:
                            # Get documents with scores for better understanding
                            docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=self.retrieval_k)
                            print(f"[INFO] Relevance scores (lower is better):")
                            for i, (doc, score) in enumerate(docs_with_scores[:3]):
                                preview = doc.page_content[:100].replace('\n', ' ')
                                print(f"  Doc {i+1}: score={score:.3f} - {preview}...")
                        except Exception as e:
                            print(f"[yellow]⚠ Could not get relevance scores: {str(e)}[/yellow]")
                        
                        print(f"[INFO] QA chain retrieved {len(source_docs)} documents (retrieval_k={self.retrieval_k})")
                        
                        # Convert response to string if it's an object
                        if hasattr(answer_text, 'content'):
                            answer_text = answer_text.content
                        else:
                            answer_text = str(answer_text)
                        
                        return {"response": answer_text, "context": source_docs}
                    
                    # Fallback to manual retrieval if no qa_chain
                    # Use expanded retriever if query expansion is enabled
                    retriever = self._create_expanded_retriever(self.retrieval_k)

                    # Get relevant documents - using the newer invoke method
                    try:
                        relevant_docs = retriever.invoke(question)
                    except Exception as e:
                        # Fallback to old method if invoke fails
                        print(f"[yellow]⚠ Retriever invoke failed, using fallback: {str(e)}[/yellow]")
                        relevant_docs = retriever.get_relevant_documents(question)
                    

                    if not relevant_docs:
                        return {"response": "I couldn't find any relevant information in the documents to answer your question.", "context": []}
                    
                    # Apply reranking if enabled
                    if self.reranker and relevant_docs:
                        relevant_docs = self._rerank_documents(question, relevant_docs)

                    # Format context
                    context_parts = []
                    # Use all documents after reranking (reranker already limits to reranker_top_k)
                    for i, doc in enumerate(relevant_docs):
                        context_parts.append(f"Document {i+1}:\n{doc.page_content}")
                    context = "\n\n".join(context_parts)

                    # Create prompt - load from system_prompt.md if available
                    system_prompt_file = Path("system_prompt.md")
                    if system_prompt_file.exists():
                        try:
                            with open(system_prompt_file, 'r', encoding='utf-8') as f:
                                system_prompt_template = f.read()
                            # Replace context placeholder
                            system_prompt_with_context = system_prompt_template.replace("{context}", context)
                            prompt = f"{system_prompt_with_context}\n\nQuestion: {question}\n\nAnswer:"
                        except Exception as e:
                            # Fallback prompt
                            print(f"[yellow]⚠ Could not load system prompt: {str(e)}[/yellow]")
                            prompt = f"""You are a helpful assistant. Based on the following context, please provide a comprehensive answer to the question. If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
                    else:
                        # Default fallback prompt without length restrictions
                        prompt = f"""You are a helpful assistant. Based on the following context, please provide a comprehensive answer to the question. If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

                    # Query LLM with timeout protection
                    response = self.llm_manager.invoke(prompt)
                    
                    # Clean up
                    gc.collect()

                    # Return both response and context for display
                    return {"response": response, "context": relevant_docs}

                except Exception as e:
                    # Handle specific errors
                    error_str = str(e)
                    print(f"[WARNING] Query exception: {error_str[:200]}")
                    
                    if "fds_to_keep" in error_str:
                        # Try a minimal query without retriever
                        print("[WARNING] Falling back to simple query without retrieval due to file descriptor error")
                        try:
                            simple_prompt = f"Question: {question}\n\nPlease provide a helpful response based on general knowledge."
                            response = self.llm_manager.invoke(simple_prompt)
                            return {"response": response, "context": []}
                        except Exception:
                            return {"response": "System resource error. Please restart the RAG system (Ctrl+Shift+R).", "context": []}
                    else:
                        raise e

            # Execute in thread pool
            result = await loop.run_in_executor(None, simple_query)
            return result

        except Exception as e:
            error_msg = str(e)

            # Get current provider for provider-specific error messages
            current_provider = "ollama"  # Default
            if self.settings_manager:
                current_provider = self.settings_manager.get("llm_provider", "ollama")

            # Provide specific guidance for common errors
            if "fds_to_keep" in error_msg or "Bad file descriptor" in error_msg:
                return {"response": (
                    "I encountered a system resource error. Please try:\n"
                    "1. Press Ctrl+Shift+R to restart the RAG system\n"
                    "2. Reduce chunk size in settings (Ctrl+S)\n"
                    "3. Restart the application"
                ), "context": []}
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                # Provider-specific connection error messages
                if current_provider == "ollama":
                    return {"response": (
                        "Cannot connect to Ollama. You can:\n"
                        "1. Start Ollama (run 'ollama serve' in terminal)\n"
                        "2. Install the model (run 'ollama pull llama3.2')\n"
                        "3. Or switch to an API provider in Settings (Ctrl+S)"
                    ), "context": []}
                elif current_provider == "openai":
                    return {"response": (
                        "Cannot connect to OpenAI API. Please check:\n"
                        "1. Your API key is correct in Settings (Ctrl+S)\n"
                        "2. Your internet connection is working\n"
                        "3. The API endpoint URL is correct (if using custom endpoint)\n"
                        "4. Or switch to Ollama in Settings (Ctrl+S)"
                    ), "context": []}
                elif current_provider == "anthropic":
                    return {"response": (
                        "Cannot connect to Anthropic API. Please check:\n"
                        "1. Your API key is correct in Settings (Ctrl+S)\n"
                        "2. Your internet connection is working\n"
                        "3. Or switch to Ollama in Settings (Ctrl+S)"
                    ), "context": []}
                else:
                    return {"response": (
                        "Connection error. Please check your network connection and try again.\n"
                        "You can also switch providers in Settings (Ctrl+S)"
                    ), "context": []}
            elif "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                if current_provider in ["openai", "anthropic"]:
                    return {"response": (
                        f"API authentication error for {current_provider.title()}. Please:\n"
                        "1. Check your API key in Settings (Ctrl+S)\n"
                        "2. Ensure your API key is valid and has sufficient credits\n"
                        "3. Or switch to Ollama in Settings (Ctrl+S)"
                    ), "context": []}
                else:
                    return {"response": f"Authentication error: {error_msg}", "context": []}
            elif "ollama" in error_msg.lower():
                # Specific Ollama errors even when using other providers
                return {"response": (
                    "Ollama-related error detected. You can:\n"
                    "1. Switch to an API provider in Settings (Ctrl+S)\n"
                    "2. Or fix Ollama: start service and install models"
                ), "context": []}
            else:
                return {"response": f"Error: {error_msg}", "context": []}

    def get_stats(self):
        """Get database statistics"""
        stats = self.vectorstore_manager.get_stats()
        if stats:
            # Add model and temperature info
            stats["model"] = self.llm_manager.get_current_model_name()
            stats["temperature"] = self.temperature
            stats["retrieval_k"] = self.retrieval_k
        return stats
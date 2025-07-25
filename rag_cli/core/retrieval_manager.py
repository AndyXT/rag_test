"""Retrieval management logic separated from RAGSystem for better modularity."""

from typing import List, Optional, Any
from langchain.schema import Document, BaseRetriever

from rag_cli.utils.logger import RichLogger


class ExpandedRetriever(BaseRetriever):
    """Custom retriever that performs query expansion for better results."""

    def __init__(self, query_expander, base_retriever, k: int, **kwargs):
        """
        Initialize ExpandedRetriever.

        Args:
            query_expander: Function to expand queries
            base_retriever: Base retriever to use
            k: Number of documents to retrieve
            **kwargs: Additional arguments for BaseRetriever
        """
        super().__init__(**kwargs)
        self.query_expander = query_expander
        self.base_retriever = base_retriever
        self.k = k

    def _get_relevant_documents(self, query: str, run_manager=None) -> List[Document]:
        """Get relevant documents using query expansion."""
        # Expand the query
        expanded_queries = self.query_expander(query)

        # Retrieve documents for each query
        all_docs = []
        doc_contents_seen = set()
        max_docs_per_query = 10  # Limit docs per query
        max_total_docs = 30  # Hard limit on total docs

        for expanded_query in expanded_queries:
            if len(all_docs) >= max_total_docs:
                RichLogger.warning(
                    f"Reached maximum document limit ({max_total_docs}), stopping retrieval"
                )
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
                RichLogger.info(
                    f"Query '{expanded_query[:50]}...' added {added_count} new documents"
                )
            except Exception as e:
                RichLogger.warning(
                    f"Failed to retrieve for query: {expanded_query[:50]}... - {str(e)}"
                )

        RichLogger.info(
            f"Query expansion retrieved {len(all_docs)} unique documents from {len(expanded_queries)} queries"
        )

        # Return requested number of documents
        return all_docs[: self.k]

    async def _aget_relevant_documents(self, query: str, run_manager=None) -> List[Document]:
        """Async version - just calls sync version."""
        return self._get_relevant_documents(query)


class RetrievalManager:
    """Manages document retrieval strategies and configurations."""

    def __init__(self, vectorstore_manager, query_processor):
        """
        Initialize RetrievalManager.

        Args:
            vectorstore_manager: VectorStore manager instance
            query_processor: Query processor instance
        """
        self.vectorstore_manager = vectorstore_manager
        self.query_processor = query_processor

    def create_expanded_retriever(
        self, k: int, query_expansion_llm: Optional[Any] = None
    ) -> BaseRetriever:
        """
        Create a retriever that uses query expansion if enabled.

        Args:
            k: Number of documents to retrieve
            query_expansion_llm: Optional LLM for query expansion

        Returns:
            Retriever instance (expanded or basic)
        """
        vectorstore = self.vectorstore_manager.get_vectorstore()
        if not vectorstore:
            raise ValueError("Vectorstore not initialized")

        base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        if not query_expansion_llm:
            return base_retriever

        # Create query expander function
        def query_expander(query: str) -> List[str]:
            return self.query_processor.expand_query(query, query_expansion_llm)

        return ExpandedRetriever(
            query_expander=query_expander, base_retriever=base_retriever, k=k
        )

    def log_relevance_scores(self, question: str, k: int = 3) -> None:
        """
        Log document relevance scores for debugging.

        Args:
            question: Query to get scores for
            k: Number of top documents to log
        """
        try:
            vectorstore = self.vectorstore_manager.get_vectorstore()
            if not vectorstore:
                return

            docs_with_scores = vectorstore.similarity_search_with_score(question, k=k)
            # Convert to documents with metadata
            documents = []
            for doc, score in docs_with_scores:
                doc.metadata = doc.metadata or {}
                doc.metadata["score"] = score
                documents.append(doc)
            # Use query processor to log scores
            self.query_processor.log_relevance_scores(documents)
        except Exception as e:
            RichLogger.warning(f"Could not get relevance scores: {str(e)}")

    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents using the reranker if available.

        Args:
            query: Query string
            documents: Documents to rerank

        Returns:
            Reranked documents
        """
        reranker = self.vectorstore_manager.reranker
        if reranker and documents:
            return self.vectorstore_manager.rerank_documents(query, documents)
        return documents

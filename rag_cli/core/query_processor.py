"""Query processing utilities for RAG system"""

from typing import List, Optional, Any
from langchain_core.documents import Document

from rag_cli.utils.logger import RichLogger


class QueryProcessor:
    """Handles query expansion, reranking, and processing"""

    def __init__(self, settings_manager: Any = None):
        self.settings_manager = settings_manager

    def expand_query(self, query: str, expansion_llm: Any) -> str:
        """
        Expand a query using an LLM for better retrieval

        Args:
            query: Original query
            expansion_llm: LLM to use for expansion

        Returns:
            Expanded query
        """
        if not expansion_llm:
            return query

        try:
            expansion_prompt = f"""You are a helpful assistant that expands queries to improve document retrieval.
Given a query, expand it by adding relevant keywords, synonyms, and related concepts.
Keep the expansion concise and relevant.

Original query: {query}
Expanded query:"""

            expanded = expansion_llm.invoke(expansion_prompt).strip()

            # Combine original and expanded query
            if expanded and expanded != query:
                combined = f"{query} {expanded}"
                RichLogger.info(f"Query expanded from '{query}' to '{combined}'")
                return combined

            return query

        except Exception as e:
            RichLogger.warning(f"Query expansion failed: {str(e)}")
            return query

    def refine_query(self, query: str, refinement_llm: Any) -> str:
        """
        Refine a query using an LLM to improve clarity and retrieval effectiveness

        Args:
            query: Original user query
            refinement_llm: LLM to use for refinement

        Returns:
            Refined query
        """
        if not refinement_llm:
            return query

        try:
            refinement_prompt = f"""You are an expert at reformulating questions for better information retrieval.
Your task is to take a user's question and rewrite it to be more clear, specific, and effective for searching technical documentation.

Guidelines:
- Keep the core intent of the question
- Add relevant technical terms if they seem implied
- Make vague questions more specific
- Fix grammar or clarity issues
- Keep it concise (one sentence if possible)
- Don't add unnecessary context

User's question: {query}
Improved question:"""

            refined = refinement_llm.invoke(refinement_prompt).strip()

            # Use refined query if it's meaningfully different
            if refined and refined != query and len(refined) > 5:
                RichLogger.info(f"Query refined from '{query}' to '{refined}'")
                return refined

            return query

        except Exception as e:
            RichLogger.warning(f"Query refinement failed: {str(e)}")
            return query

    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a context string

        Args:
            documents: List of retrieved documents

        Returns:
            Formatted context string
        """
        if not documents:
            return ""

        # Use list comprehension to build context parts more efficiently
        def format_document(i: int, doc: Document) -> List[str]:
            parts = [f"[Document {i + 1}]", doc.page_content.strip()]
            
            # Add metadata if available
            if hasattr(doc, "metadata") and doc.metadata:
                metadata_str = self._format_metadata(doc.metadata)
                if metadata_str:
                    parts.append(f"Metadata: {metadata_str}")
            
            parts.append("")  # Blank line
            return parts
        
        # Flatten the list of lists into a single list
        context_parts = [
            part 
            for i, doc in enumerate(documents) 
            for part in format_document(i, doc)
        ]

        return "\n".join(context_parts).strip()

    def _format_metadata(self, metadata: dict) -> str:
        """Format document metadata for display"""
        relevant_keys = ["source", "page", "chunk_id"]
        formatted = []

        for key in relevant_keys:
            if key in metadata:
                formatted.append(f"{key}={metadata[key]}")

        return ", ".join(formatted)

    def create_rag_prompt(
        self, query: str, context: str, system_prompt: Optional[str] = None
    ) -> str:
        """
        Create a prompt for RAG with context

        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional system prompt

        Returns:
            Formatted prompt
        """
        if not system_prompt:
            system_prompt = """You are a helpful AI assistant. Use the provided context to answer questions accurately.
If the answer cannot be found in the context, say so clearly."""

        prompt = f"""{system_prompt}

Context information is below:
---------------------
{context}
---------------------

Given the context information and not prior knowledge, answer the query.
Query: {query}
Answer: """

        return prompt

    def log_relevance_scores(self, documents: List[Document]) -> None:
        """
        Log relevance scores for retrieved documents

        Args:
            documents: Retrieved documents with scores
        """
        if not documents:
            return

        for i, doc in enumerate(documents):
            if hasattr(doc, "metadata") and "score" in doc.metadata:
                score = doc.metadata["score"]
                preview = doc.page_content[:100].replace("\n", " ")
                if len(doc.page_content) > 100:
                    preview += "..."

                RichLogger.debug(f"Doc {i + 1} relevance: {score:.3f} - {preview}")

    def filter_documents_by_score(
        self, documents: List[Document], min_score: float = 0.5
    ) -> List[Document]:
        """
        Filter documents by relevance score

        Args:
            documents: Documents to filter
            min_score: Minimum relevance score

        Returns:
            Filtered documents
        """
        filtered = []

        for doc in documents:
            if hasattr(doc, "metadata") and "score" in doc.metadata:
                if doc.metadata["score"] >= min_score:
                    filtered.append(doc)
            else:
                # Include documents without scores
                filtered.append(doc)

        if len(filtered) < len(documents):
            RichLogger.info(
                f"Filtered {len(documents) - len(filtered)} documents below score threshold {min_score}"
            )

        return filtered

    def deduplicate_documents(
        self, documents: List[Document], similarity_threshold: float = 0.9
    ) -> List[Document]:
        """
        Remove duplicate or highly similar documents using hash-based approach

        Args:
            documents: Documents to deduplicate
            similarity_threshold: Similarity threshold for deduplication

        Returns:
            Deduplicated documents
        """
        if len(documents) <= 1:
            return documents

        # Use a hash-based approach for exact duplicates first
        seen_hashes = set()
        unique_docs = []
        
        # First pass: remove exact duplicates using hash
        for doc in documents:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)
        
        # Second pass: remove similar documents if threshold is less than 1.0
        if similarity_threshold < 1.0 and len(unique_docs) > 1:
            # Use a more efficient approach with early termination
            final_docs = [unique_docs[0]]
            
            for doc in unique_docs[1:]:
                # Check similarity only with a sliding window of recent documents
                # This reduces comparisons while maintaining quality
                window_size = min(10, len(final_docs))  # Compare with last 10 docs max
                is_duplicate = False
                
                for i in range(len(final_docs) - 1, max(-1, len(final_docs) - window_size - 1), -1):
                    if (
                        self._calculate_similarity(
                            doc.page_content, final_docs[i].page_content
                        )
                        > similarity_threshold
                    ):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    final_docs.append(doc)
            
            unique_docs = final_docs

        if len(unique_docs) < len(documents):
            RichLogger.info(
                f"Removed {len(documents) - len(unique_docs)} duplicate documents"
            )

        return unique_docs

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (Jaccard similarity)"""
        # Convert to sets of words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 0.0

        return len(intersection) / len(union)

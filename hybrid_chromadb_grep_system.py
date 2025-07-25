#!/usr/bin/env python3
"""
Hybrid ChromaDB + Intelligent Grep System
Demonstrates how to combine vector search with intelligent grep for optimal results
"""

import tempfile
import json
import re
import subprocess
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SearchResult:
    """Unified search result from both ChromaDB and grep"""
    content: str
    source: str
    score: float
    search_type: str  # 'semantic', 'exact', 'hybrid'
    metadata: Dict = None
    line_number: Optional[int] = None

@dataclass
class HybridSearchResponse:
    """Combined response from hybrid search system"""
    semantic_results: List[SearchResult]
    grep_results: List[SearchResult]
    combined_results: List[SearchResult]
    enhancement_data: Dict
    search_strategy: str
    total_processing_time: float

class HybridSearchCoordinator:
    """
    Coordinates between ChromaDB semantic search and intelligent grep
    """
    
    def __init__(self, small_llm, main_llm, chroma_vectorstore, documents_text_path):
        self.small_llm = small_llm
        self.main_llm = main_llm
        self.chroma_vectorstore = chroma_vectorstore
        self.documents_text_path = documents_text_path
        
        # Search strategy weights
        self.strategy_weights = {
            'semantic_only': {'semantic': 1.0, 'grep': 0.0},
            'grep_only': {'semantic': 0.0, 'grep': 1.0},
            'hybrid_balanced': {'semantic': 0.5, 'grep': 0.5},
            'semantic_primary': {'semantic': 0.7, 'grep': 0.3},
            'grep_primary': {'semantic': 0.3, 'grep': 0.7},
            'intelligent_adaptive': 'auto'  # Let small LLM decide
        }
    
    def search(self, query: str, strategy: str = 'intelligent_adaptive', 
               max_results: int = 10) -> HybridSearchResponse:
        """
        Perform hybrid search using both ChromaDB and intelligent grep
        """
        import time
        start_time = time.time()
        
        # Step 1: Analyze query and determine optimal strategy
        if strategy == 'intelligent_adaptive':
            strategy = self._determine_optimal_strategy(query)
        
        # Step 2: Generate grep patterns using small LLM
        grep_patterns = self._generate_grep_patterns(query)
        
        # Step 3: Execute searches in parallel
        semantic_results = self._search_chromadb(query, max_results)
        grep_results = self._search_with_grep(grep_patterns, max_results)
        
        # Step 4: Combine and rank results
        combined_results = self._combine_results(
            semantic_results, grep_results, strategy, query
        )
        
        # Step 5: Enhance query with findings
        enhancement_data = self._enhance_query_with_findings(
            query, semantic_results, grep_results
        )
        
        processing_time = time.time() - start_time
        
        return HybridSearchResponse(
            semantic_results=semantic_results,
            grep_results=grep_results,
            combined_results=combined_results[:max_results],
            enhancement_data=enhancement_data,
            search_strategy=strategy,
            total_processing_time=processing_time
        )
    
    def _determine_optimal_strategy(self, query: str) -> str:
        """Use small LLM to determine optimal search strategy"""
        strategy_prompt = f"""Analyze this query and determine the best search strategy:

Query: "{query}"

Consider:
- Technical/code queries → favor grep for exact matches
- Conceptual questions → favor semantic search
- Mixed queries → use balanced hybrid

Return one of: semantic_primary, grep_primary, hybrid_balanced

Strategy:"""
        
        try:
            response = self.small_llm.invoke(strategy_prompt).strip().lower()
            
            if 'grep' in response or 'exact' in response:
                return 'grep_primary'
            elif 'semantic' in response or 'conceptual' in response:
                return 'semantic_primary'
            else:
                return 'hybrid_balanced'
                
        except Exception:
            return 'hybrid_balanced'  # Safe default
    
    def _generate_grep_patterns(self, query: str) -> List[str]:
        """Generate intelligent grep patterns"""
        patterns_prompt = f"""Generate 3-5 grep patterns for: "{query}"

Focus on:
- Exact technical terms
- Code patterns (fn \\w+, let \\w+, Error::)
- Key concepts that might appear literally

Return JSON array: ["pattern1", "pattern2", ...]

Patterns:"""
        
        try:
            response = self.small_llm.invoke(patterns_prompt)
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())[:5]
        except Exception:
            pass
        
        # Fallback patterns
        return self._generate_fallback_patterns(query)
    
    def _generate_fallback_patterns(self, query: str) -> List[str]:
        """Generate basic patterns when LLM fails"""
        words = query.lower().split()
        patterns = []
        
        for word in words:
            if len(word) > 3 and word not in ['what', 'how', 'when', 'where', 'why', 'does']:
                patterns.append(word)
        
        # Add code-specific patterns
        if any(term in query.lower() for term in ['function', 'fn']):
            patterns.append(r'fn \w+')
        if any(term in query.lower() for term in ['error', 'exception']):
            patterns.append('Error')
            patterns.append('Result')
        
        return patterns[:5]
    
    def _search_chromadb(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using ChromaDB semantic search"""
        try:
            # This would be your actual ChromaDB search
            # results = self.chroma_vectorstore.similarity_search_with_score(query, k=max_results)
            
            # Mock ChromaDB results for demonstration
            mock_results = [
                ("Rust implements memory safety through ownership system...", 0.85),
                ("The borrow checker prevents data races at compile time...", 0.82),
                ("Zero-cost abstractions allow high-level features...", 0.75),
                ("Error handling in Rust uses Result and Option types...", 0.70)
            ]
            
            search_results = []
            for i, (content, score) in enumerate(mock_results):
                search_results.append(SearchResult(
                    content=content,
                    source=f"semantic_search_result_{i}",
                    score=score,
                    search_type='semantic',
                    metadata={'retrieval_method': 'chromadb_similarity'}
                ))
            
            return search_results
            
        except Exception as e:
            print(f"ChromaDB search error: {e}")
            return []
    
    def _search_with_grep(self, patterns: List[str], max_results: int) -> List[SearchResult]:
        """Search using grep with generated patterns"""
        all_results = []
        
        for pattern in patterns:
            try:
                cmd = ['grep', '-i', '-n', pattern, self.documents_text_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout.strip():
                    matches = self._parse_grep_output(result.stdout)
                    
                    for line_num, line_text in matches:
                        # Calculate relevance score based on pattern specificity
                        score = self._calculate_grep_score(pattern, line_text)
                        
                        all_results.append(SearchResult(
                            content=line_text,
                            source=f"grep_pattern_{pattern}",
                            score=score,
                            search_type='exact',
                            metadata={'pattern': pattern, 'grep_match': True},
                            line_number=line_num
                        ))
                        
            except Exception as e:
                print(f"Grep error for pattern '{pattern}': {e}")
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:max_results]
    
    def _parse_grep_output(self, output: str) -> List[Tuple[int, str]]:
        """Parse grep output into line numbers and text"""
        matches = []
        for line in output.strip().split('\n'):
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2 and parts[0].isdigit():
                    line_num = int(parts[0])
                    line_text = parts[1].strip()
                    matches.append((line_num, line_text))
        return matches
    
    def _calculate_grep_score(self, pattern: str, text: str) -> float:
        """Calculate relevance score for grep matches"""
        # Base score
        score = 0.5
        
        # Boost for longer patterns (more specific)
        score += len(pattern) / 50.0
        
        # Boost for exact case matches
        if pattern in text:
            score += 0.2
        
        # Boost for pattern frequency in text
        pattern_count = text.lower().count(pattern.lower())
        score += min(pattern_count * 0.1, 0.3)
        
        return min(score, 1.0)
    
    def _combine_results(self, semantic_results: List[SearchResult], 
                        grep_results: List[SearchResult], strategy: str, 
                        query: str) -> List[SearchResult]:
        """Combine and rank results from both search methods"""
        
        weights = self.strategy_weights.get(strategy, self.strategy_weights['hybrid_balanced'])
        
        # Apply strategy weights
        for result in semantic_results:
            result.score *= weights['semantic']
            
        for result in grep_results:
            result.score *= weights['grep']
        
        # Combine all results
        all_results = semantic_results + grep_results
        
        # Remove duplicates based on content similarity
        unique_results = self._deduplicate_results(all_results)
        
        # Sort by combined score
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        # Boost results that appear in both searches
        boosted_results = self._boost_cross_method_matches(unique_results)
        
        return boosted_results
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on content similarity"""
        unique_results = []
        seen_content = set()
        
        for result in results:
            # Simple deduplication based on first 50 characters
            content_key = result.content[:50].lower().strip()
            
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)
        
        return unique_results
    
    def _boost_cross_method_matches(self, results: List[SearchResult]) -> List[SearchResult]:
        """Boost results that were found by both semantic and grep search"""
        semantic_content = {r.content[:50].lower() for r in results if r.search_type == 'semantic'}
        grep_content = {r.content[:50].lower() for r in results if r.search_type == 'exact'}
        
        cross_matches = semantic_content.intersection(grep_content)
        
        for result in results:
            content_key = result.content[:50].lower()
            if content_key in cross_matches:
                result.score += 0.3  # Boost for cross-method validation
                result.search_type = 'hybrid'  # Mark as validated by both methods
        
        # Re-sort after boosting
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def _enhance_query_with_findings(self, original_query: str, 
                                   semantic_results: List[SearchResult],
                                   grep_results: List[SearchResult]) -> Dict:
        """Enhance original query using findings from both search methods"""
        
        # Collect key findings
        semantic_concepts = [r.content[:100] for r in semantic_results[:3]]
        exact_matches = [r.content[:100] for r in grep_results[:3]]
        
        context_summary = f"""
Semantic findings: {len(semantic_results)} results
Exact matches: {len(grep_results)} results
Key concepts: {', '.join(semantic_concepts[:2])}
Exact patterns: {', '.join([r.metadata.get('pattern', '') for r in grep_results[:3] if r.metadata])}
"""
        
        enhancement_prompt = f"""Enhance this query using both semantic and exact search findings:

Original: "{original_query}"

Findings:
{context_summary}

Create an enhanced query that incorporates both conceptual understanding and specific technical details.

Enhanced query:"""
        
        try:
            enhanced_query = self.small_llm.invoke(enhancement_prompt).strip()
            
            return {
                'original_query': original_query,
                'enhanced_query': enhanced_query,
                'semantic_results_count': len(semantic_results),
                'grep_results_count': len(grep_results),
                'context_summary': context_summary.strip(),
                'confidence_score': min((len(semantic_results) + len(grep_results)) / 20.0, 1.0)
            }
            
        except Exception as e:
            return {
                'original_query': original_query,
                'enhanced_query': original_query,
                'semantic_results_count': len(semantic_results),
                'grep_results_count': len(grep_results),
                'context_summary': 'Enhancement failed',
                'confidence_score': 0.3
            }

class IntegratedRAGSystem:
    """
    Complete RAG system integrating ChromaDB, intelligent grep, and LLMs
    """
    
    def __init__(self, small_llm, main_llm, chroma_vectorstore, documents_text_path):
        self.hybrid_coordinator = HybridSearchCoordinator(
            small_llm, main_llm, chroma_vectorstore, documents_text_path
        )
        self.main_llm = main_llm
    
    def query(self, user_query: str, search_strategy: str = 'intelligent_adaptive') -> Dict:
        """
        Process query using integrated ChromaDB + Grep system
        """
        
        # Step 1: Perform hybrid search
        search_response = self.hybrid_coordinator.search(user_query, search_strategy)
        
        # Step 2: Build context from combined results
        context = self._build_context_from_results(search_response.combined_results)
        
        # Step 3: Generate final response
        final_response = self._generate_final_response(
            search_response.enhancement_data['enhanced_query'],
            context,
            search_response
        )
        
        return {
            'original_query': user_query,
            'enhanced_query': search_response.enhancement_data['enhanced_query'],
            'search_strategy': search_response.search_strategy,
            'semantic_results_count': len(search_response.semantic_results),
            'grep_results_count': len(search_response.grep_results),
            'combined_results_count': len(search_response.combined_results),
            'processing_time': search_response.total_processing_time,
            'final_response': final_response,
            'search_details': {
                'semantic_results': [{'content': r.content[:100], 'score': r.score} 
                                   for r in search_response.semantic_results[:3]],
                'grep_results': [{'content': r.content[:100], 'score': r.score, 'pattern': r.metadata.get('pattern', '')} 
                               for r in search_response.grep_results[:3]],
                'top_combined': [{'content': r.content[:100], 'score': r.score, 'type': r.search_type} 
                               for r in search_response.combined_results[:5]]
            }
        }
    
    def _build_context_from_results(self, results: List[SearchResult]) -> str:
        """Build context string from search results"""
        context_parts = []
        
        for i, result in enumerate(results[:5], 1):
            context_parts.append(f"\nResult {i} ({result.search_type}):")
            context_parts.append(f"Score: {result.score:.3f}")
            context_parts.append(f"Content: {result.content}")
            
            if result.metadata:
                if result.metadata.get('pattern'):
                    context_parts.append(f"Pattern: {result.metadata['pattern']}")
                if result.line_number:
                    context_parts.append(f"Line: {result.line_number}")
        
        return "\n".join(context_parts)
    
    def _generate_final_response(self, enhanced_query: str, context: str, 
                               search_response: HybridSearchResponse) -> str:
        """Generate final response using main LLM"""
        
        final_prompt = f"""Answer this question using both semantic understanding and exact technical details:

Enhanced Question: {enhanced_query}

Search Results (combining semantic and exact matches):
{context}

Search Strategy Used: {search_response.search_strategy}
Total Results: {len(search_response.combined_results)} (Semantic: {len(search_response.semantic_results)}, Exact: {len(search_response.grep_results)})

Provide a comprehensive answer that leverages both the conceptual understanding from semantic search and the precise technical details from exact pattern matching.

Answer:"""
        
        return self.main_llm.invoke(final_prompt)

# Mock classes for demonstration
class MockSmallLLM:
    def invoke(self, prompt: str) -> str:
        if "search strategy" in prompt.lower():
            if any(term in prompt.lower() for term in ['function', 'fn', 'code', 'syntax']):
                return "grep_primary"
            elif any(term in prompt.lower() for term in ['concept', 'how', 'why', 'explain']):
                return "semantic_primary"
            else:
                return "hybrid_balanced"
        
        elif "grep patterns" in prompt.lower():
            if "memory" in prompt.lower():
                return '["memory safety", "ownership", "borrow checker", "fn \\\\w+"]'
            elif "function" in prompt.lower():
                return '["fn \\\\w+", "function", "def"]'
            elif "error" in prompt.lower():
                return '["Error", "Result", "Err", "match"]'
            else:
                return '["rust", "let \\\\w+", "fn \\\\w+"]'
        
        elif "enhance" in prompt.lower():
            if "memory" in prompt.lower():
                return "How does Rust implement memory safety through ownership, borrowing, and compile-time checking, with specific examples from the codebase?"
            else:
                return "Enhanced query incorporating both semantic context and exact technical patterns found in the documentation"
        
        return "Mock LLM response"

class MockMainLLM:
    def invoke(self, prompt: str) -> str:
        return """Based on the hybrid search combining semantic understanding and exact pattern matching:

**Semantic Analysis** provides conceptual understanding and context
**Exact Pattern Matching** finds specific code examples and technical details

This dual approach ensures both comprehensive coverage and precise technical accuracy. The search strategy automatically adapts based on query type:
- Technical queries favor exact matching for code patterns
- Conceptual queries favor semantic search for understanding  
- Mixed queries use balanced hybrid approach

The combined results are ranked and deduplicated, with cross-validated findings receiving priority scoring."""

def demonstrate_hybrid_system():
    """Demonstrate the hybrid ChromaDB + Grep system"""
    
    print("🚀 Hybrid ChromaDB + Intelligent Grep Demo")
    print("="*60)
    
    # Setup mock text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        tmp_file.write("""
fn main() {
    let s = String::from("hello");
    println!("Memory safety through ownership");
}

Error handling in Rust uses Result<T, E>
The borrow checker prevents data races
""")
        tmp_file_path = tmp_file.name
    
    try:
        # Initialize system
        small_llm = MockSmallLLM()
        main_llm = MockMainLLM()
        chroma_vectorstore = None  # Mock ChromaDB
        
        integrated_system = IntegratedRAGSystem(
            small_llm, main_llm, chroma_vectorstore, tmp_file_path
        )
        
        # Test different query types
        test_queries = [
            ("How does Rust handle memory safety?", "intelligent_adaptive"),
            ("Show me function syntax examples", "grep_primary"),
            ("What are the benefits of Rust?", "semantic_primary"),
            ("Error handling patterns in Rust", "hybrid_balanced")
        ]
        
        for query, strategy in test_queries:
            print(f"\n🎯 Query: '{query}' (Strategy: {strategy})")
            print("-" * 50)
            
            result = integrated_system.query(query, strategy)
            
            print(f"Enhanced Query: {result['enhanced_query']}")
            print(f"Search Strategy: {result['search_strategy']}")
            print(f"Results: Semantic={result['semantic_results_count']}, "
                  f"Grep={result['grep_results_count']}, "
                  f"Combined={result['combined_results_count']}")
            print(f"Processing Time: {result['processing_time']:.3f}s")
            
            print("\nTop Combined Results:")
            for i, res in enumerate(result['search_details']['top_combined'][:3], 1):
                print(f"  {i}. [{res['type']}] Score: {res['score']:.3f}")
                print(f"     {res['content']}")
            
            print(f"\nFinal Response:\n{result['final_response']}")
            print("\n" + "="*60)
    
    finally:
        import os
        os.unlink(tmp_file_path)

if __name__ == "__main__":
    demonstrate_hybrid_system()
#!/usr/bin/env python3
"""
Intelligent Grep Coordinator System
Uses a smaller LLM to analyze queries, perform targeted grep searches,
and enhance the query with relevant context before sending to the main model.
"""

import re
import subprocess
import tempfile
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class GrepResult:
    """Represents a grep search result"""
    pattern: str
    matches: List[Tuple[int, str]]
    source_context: str
    relevance_score: float = 0.0

@dataclass
class EnhancedQuery:
    """Enhanced query with context from grep results"""
    original_query: str
    enhanced_query: str
    grep_results: List[GrepResult]
    context_summary: str
    confidence_score: float

class SmallLLMGrepCoordinator:
    """
    Uses a small LLM to coordinate grep searches and enhance queries
    """
    
    def __init__(self, small_model, main_model, documents_text_path=None):
        """
        Initialize the coordinator
        
        Args:
            small_model: Small, fast LLM for grep coordination (e.g., llama3.2:3b)
            main_model: Main LLM for final response generation
            documents_text_path: Path to extracted document text
        """
        self.small_model = small_model
        self.main_model = main_model
        self.documents_text_path = documents_text_path
        self.grep_patterns_cache = {}
        
    def process_query(self, user_query: str) -> EnhancedQuery:
        """
        Main pipeline: analyze query -> grep -> enhance -> return
        """
        # Step 1: Analyze query and generate grep patterns
        patterns = self._generate_grep_patterns(user_query)
        
        # Step 2: Execute grep searches
        grep_results = self._execute_grep_searches(patterns)
        
        # Step 3: Analyze results and enhance query
        enhanced_query = self._enhance_query_with_context(user_query, grep_results)
        
        return enhanced_query
    
    def _generate_grep_patterns(self, query: str) -> List[str]:
        """
        Use small LLM to analyze the query and generate relevant grep patterns
        """
        analysis_prompt = f"""You are a grep pattern generator. Analyze this user query and generate 3-5 specific grep patterns that would find relevant information in technical documents.

User Query: "{query}"

Consider:
1. Key technical terms that should be searched exactly
2. Function names, variable names, or code patterns
3. Error messages or specific terminology
4. Related concepts that might provide context

Generate patterns that would be used with grep -i (case insensitive). Return ONLY a JSON list of strings, no explanation.

Example: ["memory safety", "fn \\w+", "Error::", "ownership", "borrow checker"]

Patterns:"""

        try:
            response = self.small_model.invoke(analysis_prompt)
            
            # Extract JSON from response
            patterns = self._extract_json_list(response)
            
            # Add some default patterns based on query analysis
            patterns.extend(self._generate_fallback_patterns(query))
            
            return list(set(patterns))  # Remove duplicates
            
        except Exception as e:
            print(f"Error generating patterns: {e}")
            return self._generate_fallback_patterns(query)
    
    def _extract_json_list(self, response: str) -> List[str]:
        """Extract JSON list from LLM response"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Fallback: extract patterns from lines
            lines = response.strip().split('\n')
            patterns = []
            for line in lines:
                line = line.strip().strip('"\'').strip(',')
                if line and not line.startswith('#'):
                    patterns.append(line)
            return patterns[:5]  # Limit to 5 patterns
            
        except Exception:
            return []
    
    def _generate_fallback_patterns(self, query: str) -> List[str]:
        """Generate basic patterns when LLM analysis fails"""
        words = query.lower().split()
        patterns = []
        
        # Add individual important words
        for word in words:
            if len(word) > 3 and word not in ['what', 'how', 'when', 'where', 'why']:
                patterns.append(word)
        
        # Add common technical patterns if relevant
        if any(term in query.lower() for term in ['function', 'fn', 'def']):
            patterns.append(r'fn \w+')
            patterns.append(r'def \w+')
        
        if any(term in query.lower() for term in ['error', 'exception']):
            patterns.append('Error')
            patterns.append('Exception')
        
        return patterns[:3]
    
    def _execute_grep_searches(self, patterns: List[str]) -> List[GrepResult]:
        """Execute grep searches for all patterns"""
        if not self.documents_text_path:
            return []
        
        results = []
        
        for pattern in patterns:
            try:
                # Execute grep command
                cmd = ['grep', '-i', '-n', pattern, self.documents_text_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout.strip():
                    matches = self._parse_grep_output(result.stdout)
                    
                    if matches:
                        grep_result = GrepResult(
                            pattern=pattern,
                            matches=matches,
                            source_context=self._extract_context_around_matches(matches),
                            relevance_score=self._calculate_relevance_score(pattern, matches)
                        )
                        results.append(grep_result)
                
            except Exception as e:
                print(f"Error executing grep for pattern '{pattern}': {e}")
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:10]  # Return top 10 results
    
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
    
    def _extract_context_around_matches(self, matches: List[Tuple[int, str]]) -> str:
        """Extract context around matches for better understanding"""
        if not matches:
            return ""
        
        # Get sample of matches with their text
        context_parts = []
        for line_num, line_text in matches[:5]:  # Use first 5 matches
            context_parts.append(f"Line {line_num}: {line_text}")
        
        return "\n".join(context_parts)
    
    def _calculate_relevance_score(self, pattern: str, matches: List[Tuple[int, str]]) -> float:
        """Calculate relevance score for a grep result"""
        if not matches:
            return 0.0
        
        # Score based on number of matches and pattern specificity
        match_count_score = min(len(matches) / 10.0, 1.0)  # Normalize to max 1.0
        pattern_specificity = len(pattern) / 20.0  # Longer patterns are more specific
        
        return (match_count_score + pattern_specificity) / 2.0
    
    def _enhance_query_with_context(self, original_query: str, grep_results: List[GrepResult]) -> EnhancedQuery:
        """
        Use small LLM to analyze grep results and enhance the original query
        """
        if not grep_results:
            return EnhancedQuery(
                original_query=original_query,
                enhanced_query=original_query,
                grep_results=[],
                context_summary="No relevant grep results found",
                confidence_score=0.0
            )
        
        # Prepare context from grep results
        context_text = self._prepare_context_for_analysis(grep_results)
        
        enhancement_prompt = f"""You are a query enhancement specialist. Given the original user query and relevant information found through document search, enhance the query to be more specific and contextual.

Original Query: "{original_query}"

Relevant Information Found:
{context_text}

Tasks:
1. Analyze what specific information was found
2. Enhance the original query with relevant context and specificity
3. Make the query more precise while preserving the user's intent

Return a JSON response with:
{{
    "enhanced_query": "the improved query with added context",
    "context_summary": "brief summary of what information was found",
    "confidence_score": 0.8
}}

Response:"""

        try:
            response = self.small_model.invoke(enhancement_prompt)
            enhancement_data = self._extract_enhancement_json(response)
            
            return EnhancedQuery(
                original_query=original_query,
                enhanced_query=enhancement_data.get('enhanced_query', original_query),
                grep_results=grep_results,
                context_summary=enhancement_data.get('context_summary', 'Context found'),
                confidence_score=enhancement_data.get('confidence_score', 0.5)
            )
            
        except Exception as e:
            print(f"Error enhancing query: {e}")
            return EnhancedQuery(
                original_query=original_query,
                enhanced_query=original_query,
                grep_results=grep_results,
                context_summary="Grep results found but enhancement failed",
                confidence_score=0.3
            )
    
    def _prepare_context_for_analysis(self, grep_results: List[GrepResult]) -> str:
        """Prepare context text from grep results for analysis"""
        context_parts = []
        
        for i, result in enumerate(grep_results[:5], 1):  # Top 5 results
            context_parts.append(f"\nResult {i} (pattern: '{result.pattern}'):")
            context_parts.append(result.source_context)
        
        return "\n".join(context_parts)
    
    def _extract_enhancement_json(self, response: str) -> Dict:
        """Extract JSON from enhancement response"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Fallback: parse manually
            enhanced_query = original_query = ""
            context_summary = "Analysis completed"
            confidence_score = 0.5
            
            lines = response.split('\n')
            for line in lines:
                if 'enhanced_query' in line.lower():
                    enhanced_query = line.split(':', 1)[-1].strip().strip('"')
                elif 'context_summary' in line.lower():
                    context_summary = line.split(':', 1)[-1].strip().strip('"')
                elif 'confidence' in line.lower():
                    try:
                        confidence_score = float(re.findall(r'[\d.]+', line)[0])
                    except:
                        pass
            
            return {
                'enhanced_query': enhanced_query,
                'context_summary': context_summary,
                'confidence_score': confidence_score
            }
            
        except Exception:
            return {
                'enhanced_query': '',
                'context_summary': 'Enhancement failed',
                'confidence_score': 0.2
            }

class RAGWithIntelligentGrep:
    """
    Enhanced RAG system with intelligent grep coordination
    """
    
    def __init__(self, small_model, main_model, documents_text_path):
        self.grep_coordinator = SmallLLMGrepCoordinator(
            small_model, main_model, documents_text_path
        )
        self.main_model = main_model
    
    def query(self, user_query: str, use_grep_enhancement: bool = True) -> Dict:
        """
        Process query with optional grep enhancement
        """
        if use_grep_enhancement:
            # Step 1: Enhance query using grep coordinator
            enhanced_query_obj = self.grep_coordinator.process_query(user_query)
            
            # Step 2: Use enhanced query with main model
            final_prompt = self._build_final_prompt(enhanced_query_obj)
            main_response = self.main_model.invoke(final_prompt)
            
            return {
                'original_query': user_query,
                'enhanced_query': enhanced_query_obj.enhanced_query,
                'grep_results': enhanced_query_obj.grep_results,
                'context_summary': enhanced_query_obj.context_summary,
                'confidence_score': enhanced_query_obj.confidence_score,
                'final_response': main_response,
                'enhancement_used': True
            }
        else:
            # Direct query to main model
            main_response = self.main_model.invoke(user_query)
            return {
                'original_query': user_query,
                'final_response': main_response,
                'enhancement_used': False
            }
    
    def _build_final_prompt(self, enhanced_query: EnhancedQuery) -> str:
        """Build the final prompt for the main model"""
        context_info = ""
        if enhanced_query.grep_results:
            context_info = f"""

Additional Context Found:
{enhanced_query.context_summary}

Specific Information:
"""
            for result in enhanced_query.grep_results[:3]:  # Top 3 results
                context_info += f"\nPattern '{result.pattern}' found:\n{result.source_context}\n"
        
        final_prompt = f"""You are a helpful AI assistant. Answer the following question based on the available information.

Original Question: {enhanced_query.original_query}
Enhanced Context: {enhanced_query.enhanced_query}
{context_info}

Please provide a comprehensive answer using both your knowledge and the specific information found in the documents.

Answer:"""
        
        return final_prompt

# Example usage and demonstration
def demo_intelligent_grep():
    """Demonstrate the intelligent grep coordination system"""
    
    # Mock LLM classes for demonstration
    class MockSmallLLM:
        def invoke(self, prompt: str) -> str:
            if "grep pattern generator" in prompt.lower():
                if "memory safety" in prompt.lower() or "rust" in prompt.lower():
                    return '["memory safety", "ownership", "borrow checker", "fn \\\\w+", "unsafe"]'
                elif "function" in prompt.lower():
                    return '["fn \\\\w+", "def \\\\w+", "function", "method"]'
                else:
                    return '["error", "exception", "warning"]'
            
            elif "query enhancement" in prompt.lower():
                return """{
                    "enhanced_query": "How does Rust implement memory safety through ownership and borrowing, and what are the specific mechanisms?",
                    "context_summary": "Found information about ownership, borrowing, memory safety, and function definitions",
                    "confidence_score": 0.85
                }"""
            
            return "Analysis complete"
    
    class MockMainLLM:
        def invoke(self, prompt: str) -> str:
            return """Based on the context provided, Rust implements memory safety through several key mechanisms:

1. **Ownership System**: Every value has a single owner, preventing data races
2. **Borrowing**: References allow using values without taking ownership  
3. **Borrow Checker**: Compile-time verification of memory access patterns
4. **RAII**: Automatic cleanup when values go out of scope

The specific information found shows these concepts in action throughout the codebase."""
    
    # Create mock text file for demonstration
    sample_text = """
Line 1: Programming Rust: Memory Safety
Line 10: Rust implements memory safety through ownership
Line 15: fn main() {
Line 16:     let s = String::from("hello");
Line 17: }
Line 25: The borrow checker prevents data races
Line 30: unsafe blocks allow bypassing safety checks
Line 35: fn calculate_length(s: &String) -> usize {
Line 40: Error: cannot borrow as mutable
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        tmp_file.write(sample_text)
        tmp_file_path = tmp_file.name
    
    try:
        # Initialize the system
        small_llm = MockSmallLLM()
        main_llm = MockMainLLM()
        
        rag_system = RAGWithIntelligentGrep(small_llm, main_llm, tmp_file_path)
        
        # Test queries
        test_queries = [
            "How does Rust handle memory safety?",
            "What are function definitions in Rust?",
            "Show me error handling patterns"
        ]
        
        print("=== Intelligent Grep Coordination Demo ===\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"Test {i}: {query}")
            print("-" * 50)
            
            result = rag_system.query(query)
            
            print(f"Original Query: {result['original_query']}")
            print(f"Enhanced Query: {result['enhanced_query']}")
            print(f"Context Summary: {result['context_summary']}")
            print(f"Confidence Score: {result['confidence_score']}")
            print(f"Grep Results Found: {len(result['grep_results'])}")
            
            for grep_result in result['grep_results'][:2]:  # Show first 2
                print(f"  Pattern '{grep_result.pattern}': {len(grep_result.matches)} matches")
            
            print(f"\nFinal Response:\n{result['final_response']}")
            print("\n" + "="*70 + "\n")
    
    finally:
        # Cleanup
        import os
        os.unlink(tmp_file_path)

if __name__ == "__main__":
    demo_intelligent_grep()
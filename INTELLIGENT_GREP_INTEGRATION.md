# Intelligent Grep Coordinator Integration Guide

## Overview

This guide shows you how to integrate a **small LLM grep coordinator** into your existing RAG system. The coordinator acts as an intelligent intermediary that:

1. **Analyzes user queries** using a small, fast LLM
2. **Generates targeted grep patterns** for precise document search
3. **Executes multiple grep searches** to find relevant information
4. **Enhances the original query** with context from grep results
5. **Sends the enhanced query** to your main LLM for final response

## Architecture Benefits

### 🚀 Performance Benefits
- **Fast Pattern Generation**: Small LLM (e.g., llama3.2:3b) for rapid analysis
- **Targeted Search**: Grep finds exact matches quickly
- **Reduced Token Usage**: Main LLM gets pre-filtered, relevant context
- **Cost Efficiency**: Less expensive small model does the heavy lifting

### 🎯 Accuracy Benefits
- **Precise Context**: Grep finds exact technical terms and code patterns
- **Enhanced Queries**: Context-aware queries produce better responses
- **Multi-Pattern Search**: Multiple search strategies in parallel
- **Relevance Scoring**: Best matches prioritized for main model

## System Architecture

```
User Query
    ↓
Small LLM (Coordinator)
    ↓
Generate Grep Patterns: ["memory safety", "fn \\w+", "ownership"]
    ↓
Execute Multiple Greps in Parallel
    ↓
Collect & Score Results
    ↓
Small LLM (Enhancement)
    ↓
Enhanced Query + Context
    ↓
Main LLM (Final Response)
    ↓
Response to User
```

## Integration Steps

### Step 1: Add Grep Coordinator Service

Create `rag_cli/services/grep_coordinator_service.py`:

```python
import subprocess
import tempfile
import json
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class GrepResult:
    pattern: str
    matches: List[Tuple[int, str]]
    source_context: str
    relevance_score: float = 0.0

class GrepCoordinatorService:
    """Service for intelligent grep coordination"""
    
    def __init__(self, small_llm, documents_text_path):
        self.small_llm = small_llm
        self.documents_text_path = documents_text_path
        self.pattern_cache = {}
    
    def analyze_and_grep(self, query: str) -> Dict:
        """Main coordination pipeline"""
        # Step 1: Generate grep patterns
        patterns = self._generate_patterns(query)
        
        # Step 2: Execute greps
        grep_results = self._execute_greps(patterns)
        
        # Step 3: Enhance query
        enhanced_data = self._enhance_query(query, grep_results)
        
        return {
            'original_query': query,
            'patterns_used': patterns,
            'grep_results': grep_results,
            'enhanced_query': enhanced_data['enhanced_query'],
            'context_summary': enhanced_data['context_summary'],
            'confidence_score': enhanced_data['confidence_score']
        }
    
    def _generate_patterns(self, query: str) -> List[str]:
        """Generate grep patterns using small LLM"""
        prompt = f"""Generate 3-5 grep patterns for this technical query: "{query}"

Return patterns as JSON list. Focus on:
- Exact technical terms
- Function/variable patterns (fn \\w+, let \\w+)
- Error messages
- Key concepts

Example: ["memory safety", "fn \\\\w+", "Error", "ownership"]

Patterns:"""

        try:
            response = self.small_llm.invoke(prompt)
            # Extract JSON patterns
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                patterns = json.loads(json_match.group())
                return patterns[:5]  # Limit to 5
        except Exception as e:
            print(f"Pattern generation error: {e}")
        
        # Fallback patterns
        return self._fallback_patterns(query)
    
    def _execute_greps(self, patterns: List[str]) -> List[GrepResult]:
        """Execute grep searches for all patterns"""
        results = []
        
        for pattern in patterns:
            try:
                cmd = ['grep', '-i', '-n', pattern, self.documents_text_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout.strip():
                    matches = self._parse_grep_output(result.stdout)
                    if matches:
                        grep_result = GrepResult(
                            pattern=pattern,
                            matches=matches,
                            source_context=self._extract_context(matches),
                            relevance_score=self._score_result(pattern, matches)
                        )
                        results.append(grep_result)
            except Exception as e:
                print(f"Grep error for '{pattern}': {e}")
        
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
    
    def _enhance_query(self, original_query: str, grep_results: List[GrepResult]) -> Dict:
        """Enhance query using small LLM with grep context"""
        if not grep_results:
            return {
                'enhanced_query': original_query,
                'context_summary': 'No specific context found',
                'confidence_score': 0.2
            }
        
        context = self._prepare_context(grep_results)
        
        prompt = f"""Enhance this query with found context:

Original: "{original_query}"

Context found:
{context}

Return JSON:
{{
    "enhanced_query": "improved query with context",
    "context_summary": "what was found",
    "confidence_score": 0.8
}}"""

        try:
            response = self.small_llm.invoke(prompt)
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"Enhancement error: {e}")
        
        return {
            'enhanced_query': original_query,
            'context_summary': 'Context found but enhancement failed',
            'confidence_score': 0.5
        }
```

### Step 2: Modify RAG Service

Update `rag_cli/services/rag_service.py`:

```python
from .grep_coordinator_service import GrepCoordinatorService

class RAGService:
    def __init__(self, settings_manager, llm_manager, vectorstore_manager, 
                 embeddings_manager, pdf_processor):
        # ... existing initialization ...
        
        # Add grep coordinator
        self.grep_coordinator = None
        self._setup_grep_coordinator()
    
    def _setup_grep_coordinator(self):
        """Setup grep coordinator with small LLM"""
        try:
            # Use query expansion model as small LLM
            small_llm = self.llm_manager.get_query_expansion_llm()
            
            # Setup text file path (create if needed)
            text_file_path = self._ensure_text_file_exists()
            
            if small_llm and text_file_path:
                self.grep_coordinator = GrepCoordinatorService(
                    small_llm, text_file_path
                )
        except Exception as e:
            print(f"Grep coordinator setup failed: {e}")
    
    def query_documents_enhanced(self, query: str, use_grep: bool = True) -> dict:
        """Enhanced query with intelligent grep coordination"""
        
        if use_grep and self.grep_coordinator:
            # Step 1: Grep coordination
            grep_analysis = self.grep_coordinator.analyze_and_grep(query)
            
            # Step 2: Traditional semantic search with enhanced query
            enhanced_query = grep_analysis['enhanced_query']
            semantic_results = self.query_documents(enhanced_query)
            
            # Step 3: Combine results
            return {
                'query_type': 'enhanced',
                'original_query': query,
                'enhanced_query': enhanced_query,
                'grep_patterns': grep_analysis['patterns_used'],
                'grep_results_count': len(grep_analysis['grep_results']),
                'context_summary': grep_analysis['context_summary'],
                'confidence_score': grep_analysis['confidence_score'],
                'semantic_results': semantic_results,
                'final_response': semantic_results.get('response', ''),
                'processing_time': semantic_results.get('processing_time', 0)
            }
        else:
            # Fallback to traditional query
            results = self.query_documents(query)
            results['query_type'] = 'traditional'
            return results
    
    def _ensure_text_file_exists(self) -> str:
        """Ensure extracted text file exists for grep operations"""
        text_file_path = "extracted_documents.txt"
        
        try:
            # Check if file exists and is recent
            from pathlib import Path
            import time
            
            text_file = Path(text_file_path)
            
            # Regenerate if file is older than 1 hour or doesn't exist
            if (not text_file.exists() or 
                time.time() - text_file.stat().st_mtime > 3600):
                
                self._regenerate_text_file(text_file_path)
            
            return text_file_path
            
        except Exception as e:
            print(f"Text file setup error: {e}")
            return None
    
    def _regenerate_text_file(self, output_path: str):
        """Regenerate text file from current document store"""
        try:
            # Get all documents from vectorstore
            # This is a simplified version - adapt to your vectorstore
            documents = self._get_all_documents_from_vectorstore()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, doc in enumerate(documents):
                    # Add document separator with metadata
                    f.write(f"\n--- Document {i+1} ---\n")
                    if hasattr(doc, 'metadata') and doc.metadata:
                        source = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page', '')
                        f.write(f"Source: {source}")
                        if page:
                            f.write(f", Page: {page}")
                        f.write("\n")
                    
                    # Add content
                    f.write(doc.page_content)
                    f.write("\n")
                    
        except Exception as e:
            print(f"Text file regeneration error: {e}")
```

### Step 3: Update Query Processor

Enhance `rag_cli/core/query_processor.py`:

```python
class QueryProcessor:
    def __init__(self, settings_manager=None):
        self.settings_manager = settings_manager
        self.grep_enabled = True  # Add setting for grep usage
    
    def process_query_with_grep(self, query: str, grep_analysis: dict) -> str:
        """Process query with grep analysis results"""
        
        if not grep_analysis or grep_analysis.get('confidence_score', 0) < 0.3:
            return query  # Use original query if low confidence
        
        # Build enhanced prompt with grep context
        enhanced_prompt = self._build_grep_enhanced_prompt(query, grep_analysis)
        return enhanced_prompt
    
    def _build_grep_enhanced_prompt(self, original_query: str, grep_analysis: dict) -> str:
        """Build enhanced prompt with grep results"""
        
        context_section = ""
        if grep_analysis.get('grep_results'):
            context_section = f"""

Based on document analysis, the following relevant information was found:
{grep_analysis['context_summary']}

Specific patterns found: {', '.join(grep_analysis['patterns_used'])}
"""

        enhanced_prompt = f"""Please answer this question with attention to the specific technical details found in the documentation:

Original Question: {original_query}
Enhanced Context: {grep_analysis.get('enhanced_query', original_query)}
{context_section}

Please provide a comprehensive answer that incorporates both the specific technical details found and general knowledge where appropriate."""

        return enhanced_prompt
```

### Step 4: Update UI Commands

Add grep-enhanced commands to `rag_cli/ui/app.py`:

```python
class TUIApp(App):
    # ... existing code ...
    
    def handle_user_input(self, message: str):
        """Handle user input with optional grep enhancement"""
        
        # Check for special commands
        if message.startswith('/grep-enhanced '):
            query = message[15:]  # Remove '/grep-enhanced '
            self._handle_grep_enhanced_query(query)
            return
        
        if message.startswith('/grep-analyze '):
            query = message[14:]  # Remove '/grep-analyze '
            self._analyze_query_patterns(query)
            return
        
        # ... existing message handling ...
    
    def _handle_grep_enhanced_query(self, query: str):
        """Handle grep-enhanced query"""
        try:
            with self.status_manager.status("Analyzing query with intelligent grep..."):
                # Use enhanced query method
                result = self.rag_service.query_documents_enhanced(query, use_grep=True)
                
                # Display results with grep analysis
                self._display_enhanced_results(result)
                
        except Exception as e:
            self.error_handler.handle_error(e, "Grep-enhanced query failed")
    
    def _analyze_query_patterns(self, query: str):
        """Analyze what grep patterns would be generated"""
        try:
            if self.rag_service.grep_coordinator:
                analysis = self.rag_service.grep_coordinator.analyze_and_grep(query)
                
                # Display analysis
                self.console.print("\n🔍 Grep Pattern Analysis", style="bold blue")
                self.console.print(f"Original Query: {analysis['original_query']}")
                self.console.print(f"Generated Patterns: {analysis['patterns_used']}")
                self.console.print(f"Results Found: {len(analysis['grep_results'])}")
                self.console.print(f"Enhanced Query: {analysis['enhanced_query']}")
                self.console.print(f"Confidence: {analysis['confidence_score']:.2f}")
                
        except Exception as e:
            self.error_handler.handle_error(e, "Pattern analysis failed")
    
    def _display_enhanced_results(self, result: dict):
        """Display enhanced query results"""
        
        # Show enhancement details
        self.console.print("\n🤖 Intelligent Grep Analysis", style="bold green")
        
        enhancement_panel = Panel(
            f"[bold]Original:[/bold] {result['original_query']}\n"
            f"[bold]Enhanced:[/bold] {result['enhanced_query']}\n"
            f"[bold]Patterns Used:[/bold] {', '.join(result['grep_patterns'])}\n"
            f"[bold]Context Found:[/bold] {result['context_summary']}\n"
            f"[bold]Confidence:[/bold] {result['confidence_score']:.2f}",
            title="Query Enhancement",
            border_style="blue"
        )
        self.console.print(enhancement_panel)
        
        # Show final response
        self.console.print("\n💬 Enhanced Response", style="bold green")
        
        response_panel = Panel(
            result['final_response'],
            title="Response",
            border_style="green"
        )
        self.console.print(response_panel)
```

### Step 5: Add Configuration Settings

Update settings to include grep coordination options:

```python
# In rag_cli/config/model_config.py

# Grep Coordinator Settings
GREP_COORDINATOR_ENABLED = True
GREP_PATTERNS_CACHE_SIZE = 100
GREP_MAX_PATTERNS_PER_QUERY = 5
GREP_RELEVANCE_THRESHOLD = 0.3
GREP_CONTEXT_LINES = 5

# Small LLM for Grep Coordination
GREP_COORDINATOR_MODEL = "llama3.2:3b"  # Use your query expansion model
GREP_ENHANCEMENT_TEMPERATURE = 0.1  # Low temperature for consistent patterns
```

## Usage Examples

### 1. Basic Grep-Enhanced Query
```bash
# Use intelligent grep coordination
/grep-enhanced How does Rust handle memory safety?

# Output shows:
# - Original query
# - Generated patterns: ["memory safety", "ownership", "borrow checker"]
# - Enhanced query with context
# - Final response incorporating specific findings
```

### 2. Pattern Analysis
```bash
# Analyze what patterns would be generated
/grep-analyze What are the common error patterns in Rust?

# Shows:
# - Generated patterns: ["Error", "Result", "panic", "unwrap"]
# - Number of matches found
# - Confidence score
```

### 3. Programming-Specific Queries
```bash
/grep-enhanced Show me function definition patterns
# Patterns: ["fn \\w+", "function", "def \\w+"]

/grep-enhanced How are errors handled?
# Patterns: ["Error", "Result", "try", "catch", "except"]

/grep-enhanced What are the configuration options?
# Patterns: ["config", "settings", "options", "parameters"]
```

## Performance Optimization

### 1. Caching Strategies
```python
class OptimizedGrepCoordinator:
    def __init__(self, small_llm, documents_text_path):
        self.pattern_cache = {}  # Cache frequent patterns
        self.result_cache = {}   # Cache grep results
        self.enhancement_cache = {}  # Cache query enhancements
    
    def get_cached_patterns(self, query_hash: str) -> List[str]:
        """Get cached patterns for similar queries"""
        return self.pattern_cache.get(query_hash, [])
    
    def cache_patterns(self, query_hash: str, patterns: List[str]):
        """Cache patterns for future use"""
        if len(self.pattern_cache) < 100:  # Limit cache size
            self.pattern_cache[query_hash] = patterns
```

### 2. Async Processing
```python
import asyncio

async def process_query_async(self, query: str) -> Dict:
    """Async version for better performance"""
    
    # Generate patterns
    patterns = await self._generate_patterns_async(query)
    
    # Execute greps in parallel
    grep_tasks = [self._execute_grep_async(pattern) for pattern in patterns]
    grep_results = await asyncio.gather(*grep_tasks)
    
    # Enhance query
    enhanced_data = await self._enhance_query_async(query, grep_results)
    
    return enhanced_data
```

### 3. Smart Pattern Selection
```python
def smart_pattern_selection(self, query: str, all_patterns: List[str]) -> List[str]:
    """Select most relevant patterns based on query analysis"""
    
    # Score patterns by relevance to query
    scored_patterns = []
    for pattern in all_patterns:
        score = self._calculate_pattern_relevance(query, pattern)
        scored_patterns.append((pattern, score))
    
    # Return top N patterns
    scored_patterns.sort(key=lambda x: x[1], reverse=True)
    return [pattern for pattern, score in scored_patterns[:5]]
```

## Benefits Summary

### 🎯 Accuracy Improvements
- **Precise Context**: Finds exact technical terms and code snippets
- **Enhanced Queries**: Context-aware queries produce better responses
- **Multi-Pattern Search**: Comprehensive coverage with parallel searches
- **Relevance Scoring**: Best matches prioritized

### ⚡ Performance Benefits
- **Fast Small LLM**: Quick pattern generation and enhancement
- **Targeted Search**: Grep finds relevant content efficiently
- **Reduced Tokens**: Main LLM gets pre-filtered context
- **Cost Efficiency**: Less expensive coordination model

### 🔧 Technical Advantages
- **Exact Pattern Matching**: Finds code patterns, error messages, specific terms
- **Semantic + Exact**: Combines semantic understanding with precise matching
- **Scalable**: Works with large document collections
- **Configurable**: Adjustable patterns, thresholds, and caching

## Integration Checklist

- [ ] Install and configure small LLM (e.g., llama3.2:3b)
- [ ] Create `GrepCoordinatorService` class
- [ ] Update `RAGService` with enhanced query method
- [ ] Modify UI to support grep-enhanced commands
- [ ] Add configuration settings for grep coordination
- [ ] Setup text file generation from document store
- [ ] Test with various query types
- [ ] Optimize caching and performance
- [ ] Monitor and tune relevance scoring

This intelligent grep coordination system gives you the best of both worlds: the semantic understanding of embeddings with the precision of exact pattern matching, all coordinated by a fast, efficient small LLM.
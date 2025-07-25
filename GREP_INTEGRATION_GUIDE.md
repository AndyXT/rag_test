# Integrating Grep-Like Search with Your RAG System

## Overview

Yes, it is absolutely possible to have your model use grep or other tools to search documents in text form! Your current setup already has most of the infrastructure needed, and this guide shows you how to extend it.

## Current System Capabilities

Your RAG CLI system already includes:

✅ **PDF Text Extraction**: Using PyPDFLoader from LangChain  
✅ **Text Processing**: Document chunking and preprocessing  
✅ **Vector Search**: Semantic search through embeddings  
✅ **Linux Environment**: grep (GNU grep 3.11) is available  
✅ **Python Environment**: Full regex support  

## Available Tools

### 1. System Tools
- **grep**: GNU grep 3.11 with PCRE2 support
- **Python regex**: Full regex library with advanced patterns
- **File system**: Read/write access for temporary text files

### 2. Current RAG Infrastructure
- **PyPDFLoader**: Extracts text from PDF documents
- **Text splitters**: Chunk documents for processing
- **Vector stores**: ChromaDB for semantic search
- **Query processing**: Advanced query expansion and filtering

## Implementation Approaches

### Approach 1: Hybrid Search (Recommended)

Combine semantic search with exact pattern matching:

```python
def hybrid_search(query, documents, use_grep=True):
    # 1. Semantic search for context
    semantic_results = vector_store.similarity_search(query, k=10)
    
    # 2. Extract text and save to temporary file
    full_text = extract_all_text(documents)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as tmp_file:
        tmp_file.write(full_text)
        tmp_file.flush()
        
        # 3. Use grep for exact pattern matching
        if use_grep:
            grep_results = grep_search(tmp_file.name, query)
            
        # 4. Combine and rank results
        return combine_results(semantic_results, grep_results)
```

### Approach 2: Text Export Pipeline

Export all documents to searchable text files:

```python
def export_documents_to_text():
    """Export all PDFs to searchable text files"""
    pdf_files = Path("documents").glob("*.pdf")
    text_dir = Path("extracted_text")
    text_dir.mkdir(exist_ok=True)
    
    for pdf_path in pdf_files:
        # Extract text using existing PyPDFLoader
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        
        # Save as text file
        text_content = "\n".join(doc.page_content for doc in documents)
        text_file = text_dir / f"{pdf_path.stem}.txt"
        text_file.write_text(text_content, encoding='utf-8')
    
    return text_dir
```

### Approach 3: Real-time Grep Integration

Add grep functionality to your existing query processor:

```python
# In rag_cli/core/query_processor.py

def grep_search_documents(self, query, case_sensitive=False):
    """Add grep search to existing query processor"""
    # Extract text from current document store
    text_content = self._get_all_document_text()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        tmp_file.write(text_content)
        tmp_file_path = tmp_file.name
    
    try:
        # Use grep for pattern matching
        cmd = ['grep', '-n', '-i' if not case_sensitive else '', query, tmp_file_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return self._parse_grep_results(result.stdout)
        else:
            return []
    finally:
        os.unlink(tmp_file_path)
```

## Practical Implementation Steps

### Step 1: Extend PDF Processor

Add text export functionality to your existing PDF processor:

```python
# In rag_cli/core/pdf_processor.py

def export_text_for_grep(self, documents, output_file=None):
    """Export processed documents as searchable text"""
    if output_file is None:
        output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        output_file_path = output_file.name
    else:
        output_file_path = output_file
        output_file = open(output_file_path, 'w', encoding='utf-8')
    
    try:
        for doc in documents:
            # Add metadata headers for context
            if hasattr(doc, 'metadata'):
                output_file.write(f"\n--- {doc.metadata.get('source', 'Unknown')} ---\n")
                if 'page' in doc.metadata:
                    output_file.write(f"Page {doc.metadata['page']}\n")
            
            # Write content
            output_file.write(doc.page_content)
            output_file.write("\n")
        
        return output_file_path
    finally:
        output_file.close()
```

### Step 2: Create Grep Search Service

Add a new service for pattern-based searching:

```python
# Create rag_cli/services/grep_service.py

import subprocess
import tempfile
import re
from typing import List, Tuple, Optional

class GrepSearchService:
    """Service for grep-based document searching"""
    
    def __init__(self, pdf_processor):
        self.pdf_processor = pdf_processor
        self._text_cache = None
        self._text_file_path = None
    
    def search_pattern(self, pattern: str, case_sensitive: bool = False, 
                      regex: bool = False) -> List[Tuple[int, str]]:
        """Search for a pattern using grep"""
        if not self._text_file_path:
            self._prepare_text_file()
        
        cmd = ['grep', '-n']
        
        if not case_sensitive:
            cmd.append('-i')
        
        if regex:
            cmd.append('-E')  # Extended regex
        
        cmd.extend([pattern, self._text_file_path])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return self._parse_grep_output(result.stdout)
            return []
        except Exception as e:
            print(f"Grep search error: {e}")
            return []
    
    def search_multiple_patterns(self, patterns: List[str], 
                                operator: str = "OR") -> List[Tuple[int, str]]:
        """Search for multiple patterns"""
        all_results = []
        
        for pattern in patterns:
            results = self.search_pattern(pattern)
            all_results.extend(results)
        
        if operator == "AND":
            # Only return lines that match all patterns
            line_counts = {}
            for line_num, line_text in all_results:
                line_counts[line_num] = line_counts.get(line_num, 0) + 1
            
            return [(line_num, line_text) for line_num, line_text in all_results 
                   if line_counts[line_num] == len(patterns)]
        
        # OR operator - return all unique matches
        unique_results = {}
        for line_num, line_text in all_results:
            unique_results[line_num] = line_text
        
        return list(unique_results.items())
    
    def _prepare_text_file(self):
        """Prepare text file for grep searching"""
        # Get documents from the PDF processor or document store
        documents = self._get_current_documents()
        self._text_file_path = self.pdf_processor.export_text_for_grep(documents)
    
    def _parse_grep_output(self, output: str) -> List[Tuple[int, str]]:
        """Parse grep output into line numbers and text"""
        results = []
        for line in output.strip().split('\n'):
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2 and parts[0].isdigit():
                    line_num = int(parts[0])
                    line_text = parts[1].strip()
                    results.append((line_num, line_text))
        return results
```

### Step 3: Integrate with RAG Service

Modify your main RAG service to include grep functionality:

```python
# In rag_cli/services/rag_service.py

from .grep_service import GrepSearchService

class RAGService:
    def __init__(self, ...):
        # ... existing initialization ...
        self.grep_service = GrepSearchService(self.pdf_processor)
    
    def hybrid_search(self, query: str, use_semantic: bool = True, 
                     use_grep: bool = True) -> dict:
        """Perform hybrid search combining semantic and grep results"""
        results = {
            'semantic_results': [],
            'grep_results': [],
            'combined_score': 0
        }
        
        if use_semantic:
            # Existing semantic search
            results['semantic_results'] = self.query_documents(query)
        
        if use_grep:
            # Grep-based search
            grep_matches = self.grep_service.search_pattern(query, case_sensitive=False)
            results['grep_results'] = grep_matches
        
        # Combine and rank results
        results['combined_score'] = self._calculate_combined_score(results)
        
        return results
    
    def search_patterns(self, patterns: List[str]) -> dict:
        """Search for specific patterns in documents"""
        return {
            'patterns': patterns,
            'results': self.grep_service.search_multiple_patterns(patterns)
        }
```

### Step 4: Add CLI Commands

Extend your CLI interface to support grep functionality:

```python
# In rag_cli/ui/app.py or create new commands

def add_grep_commands(self):
    """Add grep-specific commands to the UI"""
    
    # Command: /grep <pattern>
    if input_text.startswith('/grep '):
        pattern = input_text[6:]  # Remove '/grep '
        results = self.rag_service.grep_service.search_pattern(pattern)
        
        self.display_grep_results(results)
        return
    
    # Command: /patterns <pattern1> <pattern2> ...
    if input_text.startswith('/patterns '):
        patterns = input_text[10:].split()
        results = self.rag_service.search_patterns(patterns)
        
        self.display_pattern_results(results)
        return
    
    # Command: /hybrid <query>
    if input_text.startswith('/hybrid '):
        query = input_text[8:]
        results = self.rag_service.hybrid_search(query)
        
        self.display_hybrid_results(results)
        return

def display_grep_results(self, results):
    """Display grep search results"""
    if not results:
        self.console.print("No matches found", style="yellow")
        return
    
    self.console.print(f"Found {len(results)} matches:", style="green")
    for line_num, line_text in results[:10]:  # Show first 10
        self.console.print(f"Line {line_num}: {line_text}")
    
    if len(results) > 10:
        self.console.print(f"... and {len(results) - 10} more matches")
```

## Use Cases and Examples

### 1. Exact Code Search
```bash
# Find function definitions
/grep "fn \w+("

# Find variable declarations
/grep "let \w+ ="

# Find specific error types
/grep "Error::"
```

### 2. Multi-Pattern Search
```bash
# Find memory-related concepts
/patterns memory ownership borrowing

# Find security-related terms
/patterns security vulnerability exploit
```

### 3. Hybrid Search
```bash
# Combine semantic and exact matching
/hybrid "How does Rust handle memory safety?"
```

### 4. Advanced Regex Patterns
```python
# Email addresses
grep_service.search_pattern(r'\b\w+@\w+\.\w+\b', regex=True)

# URLs
grep_service.search_pattern(r'https?://\S+', regex=True)

# Function calls
grep_service.search_pattern(r'\w+\s*\([^)]*\)', regex=True)
```

## Performance Considerations

### 1. Text Caching
- Cache extracted text to avoid re-processing PDFs
- Update cache only when documents change
- Use file modification timestamps for cache invalidation

### 2. Incremental Updates
- Only re-extract text from modified PDFs
- Maintain separate text files per document
- Use grep with multiple files for comprehensive search

### 3. Optimization Strategies
```python
# Pre-process and index common patterns
def build_pattern_index(self):
    """Build an index of common patterns for faster searching"""
    common_patterns = ['function', 'class', 'import', 'error', 'warning']
    
    self.pattern_index = {}
    for pattern in common_patterns:
        results = self.search_pattern(pattern)
        self.pattern_index[pattern] = results

# Use ripgrep for better performance (if available)
def use_ripgrep_if_available(self):
    """Use ripgrep (rg) if available for better performance"""
    try:
        subprocess.run(['rg', '--version'], capture_output=True)
        self.grep_command = 'rg'
    except FileNotFoundError:
        self.grep_command = 'grep'
```

## Integration Benefits

### 1. Complementary Search Methods
- **Semantic Search**: Understanding context and meaning
- **Grep Search**: Exact pattern matching and code search
- **Combined**: Best of both worlds

### 2. Enhanced Query Capabilities
- Find exact function names or variable references
- Search for specific error messages or patterns
- Locate code snippets or configuration examples

### 3. Developer-Friendly Features
- Regex support for complex patterns
- Line number references for precise location
- Case-sensitive and case-insensitive options

## Conclusion

Your current setup is well-positioned for grep integration:

✅ **Infrastructure Ready**: PDF processing and text extraction  
✅ **Tools Available**: grep, Python regex, file system access  
✅ **Architecture Suitable**: Modular design allows easy extension  

The recommended approach is to implement **Approach 1 (Hybrid Search)** as it provides the most comprehensive search capabilities while leveraging your existing RAG infrastructure.

This integration will give you:
- Exact pattern matching for code and technical content
- Semantic understanding for conceptual queries  
- Flexible search options for different use cases
- Enhanced developer experience for technical documentation

Start with the basic grep integration and gradually add more advanced features based on your specific needs.
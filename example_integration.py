#!/usr/bin/env python3
"""
Practical Example: Intelligent Grep Coordinator Integration
Shows how to integrate with your existing RAG CLI system
"""

import tempfile
import os
from pathlib import Path

# Example integration with your existing system
class ExampleIntegration:
    """Example showing integration with existing RAG CLI"""
    
    def __init__(self):
        # This would be your actual components
        self.small_llm = self.MockSmallLLM()
        self.main_llm = self.MockMainLLM()
        self.setup_demo_text_file()
    
    def setup_demo_text_file(self):
        """Setup demo text file representing extracted PDF content"""
        rust_content = """
--- Document 1: Programming Rust ---
Source: Programming_Rust.pdf, Page: 1

Programming Rust: Fast, Safe Systems Development

This book will teach you to write fast and safe systems programming code in Rust.
Rust is a systems programming language that combines the performance and control 
of languages like C and C++ with the memory safety of languages like Java and Python.

--- Document 1: Programming Rust ---
Source: Programming_Rust.pdf, Page: 15

fn main() {
    let s = String::from("hello");
    println!("The string is: {}", s);
}

The ownership system in Rust ensures memory safety without garbage collection.
When s goes out of scope, the memory is automatically freed.

--- Document 1: Programming Rust ---
Source: Programming_Rust.pdf, Page: 23

Error handling in Rust uses the Result<T, E> type:

fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("Cannot divide by zero".to_string())
    } else {
        Ok(a / b)
    }
}

match divide(10.0, 2.0) {
    Ok(result) => println!("Result: {}", result),
    Err(error) => println!("Error: {}", error),
}

--- Document 2: Rust for Rustaceans ---
Source: RustforRustaceans.pdf, Page: 45

The borrow checker prevents data races at compile time by enforcing these rules:
1. At any given time, you can have either one mutable reference or any number of immutable references
2. References must always be valid

fn calculate_length(s: &String) -> usize {
    s.len()  // s is a reference, so it doesn't take ownership
}

--- Document 2: Rust for Rustaceans ---
Source: RustforRustaceans.pdf, Page: 67

Unsafe Rust allows you to:
- Dereference raw pointers
- Call unsafe functions
- Access or modify mutable static variables
- Implement unsafe traits

unsafe {
    let raw_ptr = 0x1234 as *const i32;
    println!("Value: {}", *raw_ptr);  // Dangerous!
}
"""
        
        # Create temporary file for demonstration
        self.text_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        self.text_file.write(rust_content)
        self.text_file.close()
        self.text_file_path = self.text_file.name
    
    def demo_query_enhancement(self, user_query: str):
        """Demonstrate the complete query enhancement pipeline"""
        
        print(f"🎯 Processing Query: '{user_query}'")
        print("=" * 60)
        
        # Step 1: Small LLM generates grep patterns
        print("📋 Step 1: Generating grep patterns...")
        patterns = self.generate_grep_patterns(user_query)
        print(f"Generated patterns: {patterns}")
        
        # Step 2: Execute grep searches
        print("\n🔍 Step 2: Executing grep searches...")
        grep_results = self.execute_grep_searches(patterns)
        
        for result in grep_results:
            print(f"Pattern '{result['pattern']}': {len(result['matches'])} matches")
            if result['matches']:
                print(f"  Sample: {result['matches'][0][1][:80]}...")
        
        # Step 3: Enhance query with context
        print("\n🤖 Step 3: Enhancing query with context...")
        enhanced_query_data = self.enhance_query_with_context(user_query, grep_results)
        
        print(f"Original Query: {user_query}")
        print(f"Enhanced Query: {enhanced_query_data['enhanced_query']}")
        print(f"Context Summary: {enhanced_query_data['context_summary']}")
        print(f"Confidence Score: {enhanced_query_data['confidence_score']}")
        
        # Step 4: Generate final response
        print("\n💬 Step 4: Generating final response...")
        final_response = self.generate_final_response(enhanced_query_data, grep_results)
        
        print("\nFinal Response:")
        print("-" * 40)
        print(final_response)
        print("\n" + "=" * 60 + "\n")
        
        return {
            'original_query': user_query,
            'patterns': patterns,
            'grep_results': grep_results,
            'enhanced_query': enhanced_query_data['enhanced_query'],
            'final_response': final_response
        }
    
    def generate_grep_patterns(self, query: str) -> list:
        """Use small LLM to generate grep patterns"""
        prompt = f"""Generate 3-5 grep patterns for: "{query}"
Return as JSON list. Focus on exact terms, code patterns, and key concepts.
Example: ["memory safety", "fn \\\\w+", "Error", "ownership"]"""
        
        response = self.small_llm.invoke(prompt)
        
        # Simulate pattern extraction
        if "memory" in query.lower() or "safety" in query.lower():
            return ["memory safety", "ownership", "borrow checker", "fn \\w+"]
        elif "function" in query.lower():
            return ["fn \\w+", "function", "def "]
        elif "error" in query.lower():
            return ["Error", "Result", "Err", "match"]
        else:
            return ["fn \\w+", "let ", query.split()[0] if query.split() else "rust"]
    
    def execute_grep_searches(self, patterns: list) -> list:
        """Execute grep searches for all patterns"""
        import subprocess
        
        results = []
        
        for pattern in patterns:
            try:
                # Execute grep command
                cmd = ['grep', '-i', '-n', pattern, self.text_file_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                matches = []
                if result.returncode == 0 and result.stdout.strip():
                    # Parse grep output
                    for line in result.stdout.strip().split('\n'):
                        if ':' in line:
                            parts = line.split(':', 1)
                            if len(parts) == 2 and parts[0].isdigit():
                                line_num = int(parts[0])
                                line_text = parts[1].strip()
                                matches.append((line_num, line_text))
                
                results.append({
                    'pattern': pattern,
                    'matches': matches,
                    'match_count': len(matches)
                })
                
            except Exception as e:
                print(f"Grep error for '{pattern}': {e}")
                results.append({
                    'pattern': pattern,
                    'matches': [],
                    'match_count': 0
                })
        
        return results
    
    def enhance_query_with_context(self, original_query: str, grep_results: list) -> dict:
        """Use small LLM to enhance query with grep context"""
        
        # Prepare context summary
        context_parts = []
        total_matches = 0
        
        for result in grep_results:
            if result['matches']:
                total_matches += len(result['matches'])
                context_parts.append(f"Found {len(result['matches'])} matches for '{result['pattern']}'")
                
                # Add sample matches
                for line_num, line_text in result['matches'][:2]:  # First 2 matches
                    context_parts.append(f"  Line {line_num}: {line_text}")
        
        context_summary = "\n".join(context_parts)
        
        if not context_summary:
            return {
                'enhanced_query': original_query,
                'context_summary': 'No specific context found',
                'confidence_score': 0.2
            }
        
        # Use small LLM to enhance the query
        enhancement_prompt = f"""Enhance this query with the found context:

Original Query: "{original_query}"

Context Found:
{context_summary}

Provide an enhanced query that incorporates the specific technical details found."""
        
        enhanced_response = self.small_llm.invoke(enhancement_prompt)
        
        return {
            'enhanced_query': enhanced_response,
            'context_summary': f"Found {total_matches} total matches across {len([r for r in grep_results if r['matches']])} patterns",
            'confidence_score': min(total_matches / 10.0, 0.9)  # Score based on matches found
        }
    
    def generate_final_response(self, enhanced_query_data: dict, grep_results: list) -> str:
        """Generate final response using main LLM with enhanced context"""
        
        # Build context from grep results
        context_snippets = []
        for result in grep_results:
            if result['matches']:
                context_snippets.append(f"\nRelevant information for '{result['pattern']}':")
                for line_num, line_text in result['matches'][:3]:  # Top 3 matches
                    context_snippets.append(f"- {line_text}")
        
        context_section = "\n".join(context_snippets)
        
        final_prompt = f"""Answer this question using both general knowledge and the specific information found:

Enhanced Question: {enhanced_query_data['enhanced_query']}

Specific Information Found in Documents:
{context_section}

Provide a comprehensive answer that incorporates the specific details found."""
        
        return self.main_llm.invoke(final_prompt)
    
    def cleanup(self):
        """Cleanup temporary files"""
        try:
            os.unlink(self.text_file_path)
        except:
            pass
    
    # Mock LLM classes for demonstration
    class MockSmallLLM:
        """Mock small LLM for pattern generation and enhancement"""
        def invoke(self, prompt: str) -> str:
            if "grep patterns" in prompt.lower():
                if "memory" in prompt.lower():
                    return "Generated patterns focusing on memory safety, ownership, and borrow checking concepts"
                elif "function" in prompt.lower():
                    return "Generated patterns for function definitions and declarations"
                elif "error" in prompt.lower():
                    return "Generated patterns for error handling and Result types"
                else:
                    return "Generated general Rust programming patterns"
            
            elif "enhance this query" in prompt.lower():
                if "memory" in prompt.lower():
                    return "How does Rust implement memory safety through ownership, borrowing, and the borrow checker, including specific mechanisms and compile-time guarantees?"
                elif "function" in prompt.lower():
                    return "What are the syntax and patterns for function definitions in Rust, including parameter handling and return types?"
                elif "error" in prompt.lower():
                    return "How does Rust handle errors using Result types, match expressions, and error propagation patterns?"
                else:
                    return f"Enhanced version of the query with additional technical context"
            
            return "LLM response"
    
    class MockMainLLM:
        """Mock main LLM for final response generation"""
        def invoke(self, prompt: str) -> str:
            if "memory safety" in prompt.lower():
                return """Rust implements memory safety through several key mechanisms:

1. **Ownership System**: Every value has a single owner, ensuring no double-free errors
2. **Borrowing**: References allow using values without transferring ownership
3. **Borrow Checker**: Compile-time analysis prevents data races and dangling pointers

Based on the specific code examples found:
- The `fn main()` example shows ownership transfer with `String::from("hello")`
- The `calculate_length` function demonstrates borrowing with `&String`
- Memory is automatically freed when variables go out of scope

This system eliminates common memory bugs while maintaining zero-cost abstractions."""

            elif "function" in prompt.lower():
                return """Rust function definitions use the `fn` keyword with this syntax:

```rust
fn function_name(parameters) -> return_type {
    // function body
}
```

From the examples found:
- `fn main()` - entry point function with no parameters or return
- `fn divide(a: f64, b: f64) -> Result<f64, String>` - function returning Result type
- `fn calculate_length(s: &String) -> usize` - function taking a reference

Functions can return values implicitly (last expression) or explicitly with `return`."""

            elif "error" in prompt.lower():
                return """Rust handles errors primarily through the `Result<T, E>` enum:

```rust
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

Key patterns found in the documentation:
- Functions return `Result` for operations that can fail
- `match` expressions handle both success (`Ok`) and error (`Err`) cases
- Error propagation with the `?` operator
- Custom error types using `String` or structured error enums

This approach forces explicit error handling at compile time."""

            else:
                return """Based on the documentation analysis and general knowledge, here's a comprehensive answer incorporating the specific technical details found in your Rust documentation..."""

def main():
    """Demonstrate the intelligent grep coordinator system"""
    
    print("🚀 Intelligent Grep Coordinator Demo")
    print("Simulating integration with your RAG CLI system\n")
    
    # Create integration instance
    integration = ExampleIntegration()
    
    try:
        # Test different types of queries
        test_queries = [
            "How does Rust handle memory safety?",
            "What are function definitions in Rust?", 
            "How does error handling work?",
            "Show me examples of unsafe code"
        ]
        
        for query in test_queries:
            result = integration.demo_query_enhancement(query)
            
            # Show improvement metrics
            original_words = len(result['original_query'].split())
            enhanced_words = len(result['enhanced_query'].split())
            patterns_found = len([r for r in result['grep_results'] if r['matches']])
            
            print(f"📊 Enhancement Metrics:")
            print(f"   Query length: {original_words} → {enhanced_words} words")
            print(f"   Patterns with matches: {patterns_found}/{len(result['patterns'])}")
            print(f"   Total grep matches: {sum(len(r['matches']) for r in result['grep_results'])}")
            print()
    
    finally:
        integration.cleanup()

if __name__ == "__main__":
    main()
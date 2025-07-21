#!/usr/bin/env python3
"""
Basic test to verify LangChain installation and imports.
This test verifies that all required LangChain packages are properly installed.
"""

import sys

def test_langchain_imports():
    """Test that all required LangChain components can be imported."""
    try:
        # Test core LangChain imports
        import langchain
        import langchain_core
        import langchain_community
        import langchain_text_splitters
        import langchain_openai
        
        # Test specific components
        from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaLLM
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate
        
        print("✓ All LangChain imports successful!")
        print(f"✓ LangChain version: {langchain.__version__}")
        print(f"✓ LangChain Core version: {langchain_core.__version__}")
        print(f"✓ LangChain Community version: {langchain_community.__version__}")
        
        # Some packages may not have __version__ attribute
        try:
            print(f"✓ LangChain Text Splitters version: {langchain_text_splitters.__version__}")
        except AttributeError:
            print("✓ LangChain Text Splitters: imported successfully (version not available)")
            
        try:
            print(f"✓ LangChain OpenAI version: {langchain_openai.__version__}")
        except AttributeError:
            print("✓ LangChain OpenAI: imported successfully (version not available)")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_other_dependencies():
    """Test that other required dependencies can be imported."""
    try:
        import textual
        import chromadb
        import sentence_transformers
        import pypdf
        import ollama
        
        print("✓ All other dependencies imported successfully!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing LangChain installation...")
    print("=" * 50)
    
    langchain_success = test_langchain_imports()
    print()
    
    other_deps_success = test_other_dependencies()
    print()
    
    if langchain_success and other_deps_success:
        print("🎉 All tests passed! LangChain installation is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

# RAG Chat Project Overview

## Purpose
A Python RAG (Retrieval-Augmented Generation) chat application with a sophisticated Text User Interface (TUI). It enables users to query PDF documents using semantic search and LLM-powered responses through an intuitive terminal interface.

## Tech Stack
- **Language**: Python 3.13+
- **Framework**: LangChain 0.3+ (modern imports pattern)
- **UI**: Textual TUI framework with Rich formatting
- **Vector DB**: ChromaDB
- **Embeddings**: HuggingFace (sentence-transformers)
- **LLM Providers**: Ollama (local), OpenAI API, Anthropic API
- **Document Processing**: PyPDF for PDF parsing
- **Architecture**: Service-oriented with clean separation of concerns

## Key Features
- Modern TUI with responsive layout and progress indicators
- Persistent chat history and session management
- Configurable settings panel for model parameters
- Document browser for PDF management
- Query expansion and reranking for better retrieval
- Multiple LLM provider support
- Error handling with user-friendly messages
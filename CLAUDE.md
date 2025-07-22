# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python RAG (Retrieval-Augmented Generation) chat application with a sophisticated Text User Interface (TUI). The application allows users to query PDF documents using semantic search and LLM-powered responses.

## Key Commands

### Running the Application
```bash
# Main application with full RAG functionality
python rag_cli.py

# Demo version without LangChain dependencies
python demo_tui.py
```

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Using uv package manager (if available)
uv pip install -r requirements.txt
```

### Testing
```bash
# Run individual test files
python test_pdf_processing.py
python test_chroma_config.py
python test_tui_async.py
python test_tui_create_db.py
```

### Ollama Setup (Required)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull default model
ollama pull llama3.2
```

## Architecture Overview

### Core Components

1. **RAGSystem Class** (`rag_cli.py`): Core RAG functionality
   - PDF loading and processing
   - Vector database creation with ChromaDB
   - Document retrieval and LLM chain setup
   - Uses HuggingFace embeddings and Ollama LLM

2. **RagChatApp Class** (`rag_cli.py`): Main TUI application
   - Built on Textual framework
   - Manages UI components, chat interface, and user interactions
   - Handles async operations for non-blocking UI
   - Implements modal screens for settings, help, and document browsing

3. **Key Dependencies**:
   - LangChain 0.3+ (modern import structure)
   - Textual for TUI
   - ChromaDB for vector storage
   - Ollama for local LLM inference
   - HuggingFace sentence-transformers for embeddings

### Data Flow
1. PDFs placed in `./documents/` directory
2. Create database processes PDFs into ChromaDB vectors
3. User queries trigger semantic search in vector DB
4. Retrieved chunks are passed to Ollama LLM
5. Response displayed in rich TUI format

### Important Directories
- `documents/`: PDF storage (user-provided documents)
- `chroma_db/`: Vector database (auto-created)
- `__pycache__/`: Python cache files
- Virtual environments: `venv/` or `.venv/`

### Configuration Files
- `settings.json`: Persisted user settings (model, temperature, chunk size, etc.)
- `chat_history.json`: Saved chat sessions

## Development Guidelines

### LangChain Import Pattern
The project uses LangChain 0.3+ with updated import paths:
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
```

### Testing Approach
- Test files are standalone scripts in the root directory
- Each test file can be run directly with `python test_*.py`
- Tests include progress indicators and clear pass/fail output
- No pytest or unittest framework dependency

### Error Handling
- File descriptor issues are handled with proper async/await patterns
- ChromaDB persistence directory is checked/created automatically
- Clear error messages guide users to solutions

### UI/UX Patterns
- Modal screens extend `ModalScreen` class
- Keyboard bindings defined with `Binding` objects
- Progress bars for long operations
- Rich formatting for chat messages
- Responsive layout with collapsible sidebar

## Common Issues and Fixes

### Embedding Model Errors
If you encounter errors during ChromaDB database creation related to embeddings:
1. Clear the HuggingFace cache: `rm -rf ~/.cache/huggingface/`
2. Increase file descriptor limit: `ulimit -n 4096`
3. Set environment variable: `export TOKENIZERS_PARALLELISM=false`

The embedding cache can become corrupted during interrupted downloads or version conflicts.
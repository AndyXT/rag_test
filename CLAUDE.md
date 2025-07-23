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

### Embedding Model Errors & Intermittent Context Retrieval
If you experience intermittent issues where context retrieval works sometimes but not others:

**🚀 Quick Fixes:**

1. **Press Ctrl+Shift+C in the app**: Cleans cache without restarting
2. **Run the cache manager**: `python hf_cache_manager.py --clean-locks`
3. **Restart with Ctrl+Shift+R**: Restarts the RAG system completely

**🛠️ Cache Manager Commands:**
```bash
# Check cache status
python hf_cache_manager.py --info

# Clean lock files
python hf_cache_manager.py --clean-locks

# Validate cache integrity
python hf_cache_manager.py --validate

# Fix permissions
python hf_cache_manager.py --fix-permissions

# Nuclear option: clear specific model
python hf_cache_manager.py --clear-model sentence-transformers/all-MiniLM-L6-v2
```

**💡 What causes the issue:**
- HuggingFace cache corruption (most common)
- Stale lock files from interrupted model downloads
- Multiple processes accessing the same model cache simultaneously
- Corrupted partial downloads in cache
- File permission issues in cache directory

**🛡️ Prevention (automatically handled now):**
- Automatic cache cleaning on database load
- More aggressive lock file cleanup (5 minutes instead of 30)
- Zero-byte file removal
- Environment variables set to prevent tokenizer conflicts
- Single-threaded embedding initialization

**📝 How to tell if this is your issue:**
- Context display shows 0 documents even though database is loaded
- Works after deleting ~/.cache/huggingface
- Terminal shows "[WARNING] Falling back to simple query without retrieval"
- Intermittent behavior - works sometimes, fails others

**🛡️ Database Safety Features:**
- Automatic backup creation before database recreation
- Temporary database creation (never touches existing DB until success)
- Automatic restore on failure
- Backup cleanup (keeps last 3 backups automatically)

**🔧 Backup Management:**
- List backups: `python chroma_backup_manager.py list`
- Create manual backup: `python chroma_backup_manager.py backup`
- Restore backup: `python chroma_backup_manager.py restore <backup_name>`
- Cleanup old backups: `python chroma_backup_manager.py cleanup --keep 5`

**⚠️ Legacy Solution (Only if above solutions don't work):**
1. Clear the HuggingFace cache: `rm -rf ~/.cache/huggingface/`
2. Increase file descriptor limit: `ulimit -n 4096`
3. Set environment variable: `export TOKENIZERS_PARALLELISM=false`

**🚨 CRITICAL FIX:** The system now creates backups automatically and never deletes your existing database until the new one is successfully created and validated.
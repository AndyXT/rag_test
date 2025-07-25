# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python RAG (Retrieval-Augmented Generation) chat application with a sophisticated Text User Interface (TUI) built on a clean service-oriented architecture. The application enables users to query PDF documents using semantic search and LLM-powered responses through an intuitive terminal interface.

## Key Commands

### Running the Application
```bash
# Main application entry point
python -m rag_cli.main

# Alternative direct execution
python rag_cli/main.py

# Legacy entry point (still works)
python rag_cli.py
```

### Development Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Install with pyproject.toml (recommended)
pip install -e .

# Using uv package manager
uv pip install -r requirements.txt
```

### Ollama Setup (Required for local LLM)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull default model
ollama pull llama3.2:3b

# Start Ollama service
ollama serve
```

## Architecture Overview

### Service-Oriented Design
The application follows a clean service-oriented architecture with clear separation of concerns:

```
rag_cli/
├── core/              # Core business logic
│   ├── rag_system.py         # Central RAG orchestration
│   ├── llm_manager.py        # LLM provider management (Ollama/OpenAI/Anthropic)
│   ├── vectorstore_manager.py # ChromaDB operations
│   ├── embeddings_manager.py  # HuggingFace embeddings
│   ├── cache_manager.py      # Cache management for models
│   ├── pdf_processor.py      # PDF document processing
│   ├── query_processor.py    # Query expansion and formatting
│   └── error_handler.py      # User-friendly error handling
├── services/          # Service layer (coordinates core components)
│   ├── rag_service.py        # Main service coordinator
│   ├── query_service.py      # Query processing with expansion/reranking
│   ├── database_service.py   # Database creation and loading
│   └── chat_service.py       # Chat history management
├── config/            # Configuration modules
│   ├── model_config.py       # Model defaults and constants
│   ├── system_config.py      # System-level settings
│   ├── database_config.py    # Database configurations
│   └── ui_config.py          # UI constants and keybindings
├── ui/                # User interface layer
│   ├── app.py               # Main Textual TUI application
│   └── screens/             # Modal screens for settings, help, etc.
└── utils/             # Shared utilities
    ├── logger.py            # RichLogger with TUI-aware output
    ├── constants.py         # Environment setup and imports
    └── defaults.py          # Re-exports from config modules
```

### Key Design Patterns

1. **Service Layer Pattern**: The `services/` directory provides a clean interface between the UI and core business logic. The UI only interacts with services, never directly with core components.

2. **Manager Pattern**: Each core component has a dedicated manager (LLMManager, VectorStoreManager, etc.) responsible for a single concern.

3. **Dependency Injection**: Components receive their dependencies through constructors, making them testable and loosely coupled.

4. **Configuration Modules**: Settings are organized into logical modules rather than a monolithic constants file.

### Data Flow

1. **Document Processing**:
   - PDFs in `./documents/` → PDFProcessor → chunks
   - Chunks → EmbeddingsManager → vectors
   - Vectors → VectorStoreManager → ChromaDB

2. **Query Processing**:
   - User query → QueryService → QueryProcessor (expansion)
   - Expanded queries → VectorStoreManager → document retrieval
   - Retrieved docs → reranking (if enabled) → context formatting
   - Context + query → LLMManager → response

3. **UI Interaction**:
   - User input → RAGChatApp → RAGService
   - RAGService coordinates services → response
   - Response → formatted display in TUI

### Important Components

- **RAGSystem**: Core orchestrator that manages the RAG pipeline
- **RAGService**: Service layer coordinator that the UI interacts with
- **QueryProcessor**: Handles query expansion, context formatting, and document filtering
- **ErrorHandler**: Provides structured error handling with user-friendly messages
- **RichLogger**: Logging utility that detects TUI mode to prevent duplicate output

## LangChain 0.3+ Import Pattern

The project uses modern LangChain imports:
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
```

## UI Development

### Modal Screens
New modal screens should extend `ModalScreen` from `textual.screen`:
```python
from textual.screen import ModalScreen
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static, Button

class MyScreen(ModalScreen):
    def compose(self) -> ComposeResult:
        with Container(id="my-container"):
            yield Static("Content", id="my-title")
            yield Button("Close", id="close-btn")
```

### Keyboard Bindings
Add new bindings to `RAGChatApp.BINDINGS`:
```python
Binding("ctrl+x", "action_name", "Description")
```

### Progress Feedback
Use the progress bar pattern for long operations:
```python
self.processing = True
self.update_progress("Processing...", 50)
# ... operation ...
self.processing = False
```

## Common Development Tasks

### Adding a New LLM Provider
1. Update `LLMManager._initialize_llm()` with new provider case
2. Add initialization method `_initialize_<provider>()`
3. Update `SettingsScreen` with provider-specific fields
4. Add constants to `model_config.py`

### Adding a New Manager
1. Create manager class in `core/` with single responsibility
2. Initialize in `RAGSystem.__init__()`
3. Create service methods in appropriate service class
4. Update `RAGService` to coordinate if needed

### Modifying Query Processing
1. Update `QueryProcessor` for new processing logic
2. Modify `QueryService._execute_rag_sync()` if needed
3. Add configuration options to `SettingsManager.default_settings`

## Error Handling Guidelines

When handling errors:
1. Use `ErrorHandler.handle_error()` for structured error info
2. Provide recovery suggestions in error messages
3. Log errors with `RichLogger.error()` for debugging
4. Show user-friendly messages in the UI

## Cache and Database Management

### HuggingFace Cache Issues
The app includes automatic cache management, but if issues persist:
- Use `CacheManager.clean_hf_cache_locks()` programmatically
- Implement cache cleaning in database operations
- Set `TOKENIZERS_PARALLELISM=false` in environment

### Database Backups
The system automatically creates backups before database operations:
- Backups stored with timestamp
- Automatic cleanup keeps last 3 backups
- Restore on failure is automatic

## Testing Approach

Tests are standalone scripts that can be run directly:
- No pytest/unittest dependency
- Include progress indicators
- Clear pass/fail output
- Test both success and error paths

## Performance Considerations

1. **File Descriptors**: Operations are wrapped to handle descriptor limits
2. **Memory Management**: Query expansion disabled for large models
3. **Async Operations**: UI remains responsive during long operations
4. **Cache Management**: Automatic cleanup prevents corruption

## Debugging Tips

1. **Enable debug logging**: Set `RichLogger.debug_mode = True`
2. **Check TUI mode**: Ensure `RichLogger.set_tui_mode(True)` for apps
3. **Inspect service calls**: Services log operations at info level
4. **Error details**: Check `error_info` in error responses for full context
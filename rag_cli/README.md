# RAG CLI - Refactored Structure

This directory contains the refactored RAG CLI application, organized into a clean modular structure.

## Directory Structure

```
rag_cli/
├── __init__.py              # Package initialization
├── main.py                  # Main entry point
├── core/                    # Core functionality
│   ├── __init__.py
│   ├── settings_manager.py  # Settings persistence
│   ├── chat_history.py      # Chat history management
│   └── rag_system.py        # RAG system (embeddings, LLM, retrieval)
├── ui/                      # User interface components
│   ├── __init__.py
│   ├── app.py              # Main Textual application
│   └── screens/            # Modal screens
│       ├── __init__.py
│       ├── settings_screen.py      # Settings configuration
│       ├── help_screen.py          # Help documentation
│       └── document_browser.py     # Document browsing
└── utils/                   # Utility modules
    ├── __init__.py
    └── constants.py         # Constants and environment setup
```

## Usage

### As a Module
```python
from rag_cli import main
main()
```

### Direct Execution
```bash
# Run from the project root
python -m rag_cli.main

# Or use the convenience scripts
python run_refactored.py
python rag_cli_refactored.py
```

### Importing Components
```python
# Import core components
from rag_cli.core import SettingsManager, ChatHistory, RAGSystem

# Import UI components
from rag_cli.ui import RAGChatApp
from rag_cli.ui.screens import SettingsScreen, HelpScreen, DocumentBrowserScreen

# Import utilities
from rag_cli.utils import PYPERCLIP_AVAILABLE
```

## Key Benefits of Refactored Structure

1. **Modularity**: Each class is in its own file, making it easier to maintain and test
2. **Clear Organization**: Core logic, UI, and utilities are separated
3. **Reusability**: Components can be imported and used independently
4. **Scalability**: Easy to add new features without affecting existing code
5. **Testing**: Individual modules can be tested in isolation

## Module Descriptions

### Core Modules
- `settings_manager.py`: Handles application settings persistence with JSON
- `chat_history.py`: Manages chat session history and persistence
- `rag_system.py`: The main RAG engine with embeddings, vector DB, and LLM integration

### UI Modules
- `app.py`: Main Textual application that orchestrates the entire UI
- `screens/`: Modal screens for settings, help, and document browsing

### Utilities
- `constants.py`: Environment setup and application constants

## Dependencies

The refactored code maintains all original dependencies:
- LangChain (document processing, embeddings, LLM chains)
- Textual (Terminal UI framework)
- ChromaDB (Vector database)
- HuggingFace (Embeddings)
- Ollama/OpenAI/Anthropic (LLM providers)

## Future Improvements

Consider further splitting the large RAGSystem class into:
- `LLMManager`: Handle different LLM provider initialization
- `EmbeddingManager`: Manage embedding models and operations
- `DatabaseManager`: ChromaDB operations and persistence
- `QueryProcessor`: Query expansion and reranking logic
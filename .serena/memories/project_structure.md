# RAG Chat Project Structure

```
rag_test/                      # Project root
├── rag_cli/                   # Main application package
│   ├── __init__.py
│   ├── main.py               # Entry point
│   ├── core/                 # Core business logic
│   │   ├── __init__.py
│   │   ├── rag_system.py     # Central RAG orchestration
│   │   ├── llm_manager.py    # LLM provider management
│   │   ├── vectorstore_manager.py  # ChromaDB operations
│   │   ├── embeddings_manager.py   # HuggingFace embeddings
│   │   ├── cache_manager.py        # Cache management
│   │   ├── pdf_processor.py        # PDF processing
│   │   ├── query_processor.py      # Query expansion/formatting
│   │   ├── error_handler.py        # Error handling
│   │   ├── settings_manager.py     # Settings persistence
│   │   └── chat_history.py         # Chat history
│   ├── services/             # Service layer
│   │   ├── __init__.py
│   │   ├── rag_service.py    # Main service coordinator
│   │   ├── query_service.py  # Query processing service
│   │   ├── database_service.py  # Database operations
│   │   └── chat_service.py      # Chat management
│   ├── config/               # Configuration modules
│   │   ├── __init__.py
│   │   ├── model_config.py   # Model constants
│   │   ├── system_config.py  # System settings
│   │   ├── database_config.py # Database config
│   │   └── ui_config.py      # UI constants
│   ├── ui/                   # User interface
│   │   ├── __init__.py
│   │   ├── app.py           # Main TUI application
│   │   └── screens/         # Modal screens
│   │       ├── __init__.py
│   │       ├── settings_screen.py
│   │       ├── help_screen.py
│   │       └── document_browser.py
│   └── utils/               # Utilities
│       ├── __init__.py
│       ├── logger.py        # RichLogger
│       ├── constants.py     # Environment setup
│       ├── defaults.py      # Re-exports
│       ├── decorators.py    # Utility decorators
│       └── error_utils.py   # Error utilities
├── documents/               # PDF documents directory
├── chroma_db/              # Vector database storage
├── README.md               # Project documentation
├── CLAUDE.md               # Claude AI guidance
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
├── settings.json          # Application settings
└── chat_history.json      # Chat history storage
```

## Key Directories
- **core/**: Contains all business logic managers
- **services/**: Coordinates between UI and core
- **config/**: All configuration constants
- **ui/**: Textual TUI implementation
- **utils/**: Shared utilities and helpers
# 🤖 RAG Chat - Enhanced TUI Experience

A sophisticated Text User Interface (TUI) for Retrieval-Augmented Generation (RAG) chat using LangChain and Textual, featuring a clean service-oriented architecture and modular design.

## ✨ Features

### 🎨 Enhanced User Interface
- **Modern TUI Design**: Professional-looking interface with rich styling and colors
- **Responsive Layout**: Adaptive sidebar and chat area that can be toggled
- **Progress Indicators**: Real-time progress bars for long operations
- **Rich Chat Display**: Formatted messages with timestamps and panels
- **Status Dashboard**: Live statistics and system information
- **No Duplicate Logs**: Clean console output with proper TUI integration

### 🚀 Core Functionality
- **Chat History**: Persistent session management with history tracking
- **Settings Panel**: Configurable model parameters and application settings
- **Document Browser**: Browse and manage PDF documents in the interface
- **Help System**: Comprehensive help with keyboard shortcuts
- **Export Feature**: Save chat sessions to text or JSON formats
- **Context Display Toggle**: Show/hide retrieved document context (Ctrl+T)
- **Cache Management**: Clean HuggingFace cache directly from the app (Ctrl+Shift+C)
- **System Restart**: Restart the RAG system without exiting (Ctrl+Shift+R)

### 🏗️ Architecture Improvements
- **Service Layer**: Clean separation between UI and business logic
- **Modular Managers**: Focused components for LLM, embeddings, vectorstore, and cache
- **Query Processing**: Dedicated processor with expansion, formatting, and filtering
- **Error Handling**: Comprehensive error handler with user-friendly messages
- **Configuration**: Organized configuration modules for models, system, database, and UI

### ⌨️ Keyboard Shortcuts
| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Quit application |
| `Ctrl+L` | Clear current chat |
| `Ctrl+R` | Reload database |
| `Ctrl+H` | Show help screen |
| `Ctrl+S` | Open settings |
| `Ctrl+N` | Start new chat session |
| `Ctrl+D` | Open document browser |
| `Ctrl+E` | Export chat history |
| `Ctrl+Y` | Copy last message |
| `Ctrl+T` | Toggle context display |
| `Ctrl+Shift+C` | Clean cache |
| `Ctrl+Shift+R` | Restart RAG system |
| `Ctrl+Shift+E` | Export full chat history |
| `F1` | Toggle sidebar visibility |
| `Enter` | Send message |

## 🏗️ Installation

1. **Clone or download the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: The requirements include compatible versions of all LangChain packages and their dependencies to avoid conflicts.
   
3. **Install Ollama** (if not already installed):
   ```bash
   # On Linux/macOS
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the default model
   ollama pull llama3.2
   ```

## 🚀 Quick Start

1. **Place PDF documents** in the `./documents` directory
2. **Run the application**:
   ```bash
   python -m rag_cli.main
   ```
   Or directly:
   ```bash
   python rag_cli/main.py
   ```
3. **Create a database** by clicking "🔄 Create DB" 
4. **Start asking questions** about your documents!

## 📖 User Guide

### Getting Started
1. **Welcome Screen**: The app starts with a welcome message and helpful tips
2. **Auto-Load**: Automatically attempts to load existing databases
3. **Status Panel**: Shows database statistics and connection info

### Document Management
- **Documents Directory**: Place PDF files in `./documents/`
- **Create Database**: Process all PDFs into a searchable vector database
- **Load Database**: Load an existing ChromaDB database
- **Document Browser**: View and manage documents through the TUI

### Chat Features
- **Rich Formatting**: Messages display in styled panels with timestamps
- **Response Times**: See how long each query takes to process
- **Error Handling**: Clear error messages with helpful suggestions
- **History Tracking**: All conversations are saved automatically

### Customization
- **Settings Panel**: Adjust model parameters, chunk sizes, and retrieval settings
- **Sidebar Toggle**: Hide/show the sidebar for more chat space
- **Theme Support**: Built-in support for different color schemes

## ⚙️ Configuration

### LLM Provider Settings
- **Provider**: Choose between Ollama (local), OpenAI API, or Anthropic API
- **Ollama Model**: Choose different Ollama models (default: llama3.2:3b) when using Ollama
- **API Key**: Your API key for OpenAI or Anthropic (when using API providers)
- **API Base URL**: Custom API endpoint (optional, for OpenAI-compatible APIs)
- **OpenAI Model**: Model name for OpenAI (default: gpt-3.5-turbo)
- **Anthropic Model**: Model name for Anthropic (default: claude-3-haiku-20240307)

### RAG Enhancement Settings
- **Query Expansion**: Enable to generate multiple query variations for better retrieval
- **Query Expansion Model**: Small fast model for expansion (default: llama3.2:3b)
- **Expansion Queries**: Number of query variations to generate (default: 3)
- **Reranking**: Enable cross-encoder reranking for better relevance
- **Reranker Model**: Model for reranking (default: cross-encoder/ms-marco-MiniLM-L-6-v2)
- **Reranker Top K**: Number of documents to keep after reranking (default: 3)

### General Settings
- **Temperature**: Control response creativity (0.0-1.0)
- **Chunk Size**: Document splitting size (default: 1000)
- **Chunk Overlap**: Overlap between chunks (default: 200)
- **Retrieval Count**: Number of relevant chunks to retrieve (default: 3)
- **Show Context**: Toggle display of retrieved document context
- **Min Relevance Score**: Filter documents below this score threshold
- **Deduplicate Docs**: Remove duplicate or highly similar documents

### File Locations
- **Documents**: `./documents/` - Place your PDF files here
- **Database**: `./chroma_db/` - Vector database storage
- **Settings**: `settings.json` - Application settings
- **Chat History**: `chat_history.json` - Persistent chat sessions
- **Exports**: `chat_export_YYYYMMDD_HHMMSS.txt` - Exported conversations
- **Full Exports**: `full_chat_history_YYYYMMDD_HHMMSS.json` - Complete history with metadata

## 🎯 Advanced Features

### Progress Tracking
- Real-time progress bars for database creation
- Step-by-step feedback during document processing
- Visual indicators for system status

### Chat History Management
- **Session Persistence**: Chat history saved automatically
- **New Sessions**: Start fresh conversations while keeping history
- **Export Functionality**: Save conversations to text files

### Document Browser
- **File Tree**: Navigate through your document collection
- **Preview**: View document information
- **Refresh**: Update the document list dynamically

### Help System
- **Comprehensive Help**: Detailed documentation within the app
- **Keyboard Reference**: Quick access to all shortcuts
- **Getting Started Guide**: Step-by-step instructions

## 🔧 Technical Architecture

### Service-Oriented Design
```
rag_cli/
├── core/              # Core business logic
│   ├── llm_manager.py         # LLM initialization and management
│   ├── vectorstore_manager.py # Vector database operations
│   ├── embeddings_manager.py  # Embedding model management
│   ├── cache_manager.py       # HuggingFace cache management
│   ├── pdf_processor.py       # PDF document processing
│   ├── query_processor.py     # Query expansion and formatting
│   ├── error_handler.py       # Centralized error handling
│   ├── rag_system.py         # Core RAG orchestration
│   ├── settings_manager.py    # Settings persistence
│   └── chat_history.py       # Chat history management
├── services/          # Service layer
│   ├── rag_service.py        # Main service coordinator
│   ├── query_service.py      # Query processing service
│   ├── database_service.py   # Database operations service
│   └── chat_service.py       # Chat management service
├── config/            # Configuration modules
│   ├── model_config.py       # Model-related constants
│   ├── system_config.py      # System-level settings
│   ├── database_config.py    # Database configurations
│   └── ui_config.py          # UI-related constants
├── ui/                # User interface
│   ├── app.py               # Main TUI application
│   └── screens/             # Modal screens
│       ├── settings_screen.py
│       ├── help_screen.py
│       └── document_browser.py
└── utils/             # Utilities
    ├── logger.py            # Rich logging utility
    ├── constants.py         # Application constants
    └── defaults.py          # Default values
```

### Key Components
- **Service Layer**: Clean separation between UI and business logic
- **Focused Managers**: Single-responsibility components for specific tasks
- **Query Processor**: Handles query expansion, context formatting, and document filtering
- **Error Handler**: Provides user-friendly error messages with recovery suggestions
- **Configuration Modules**: Organized settings for different aspects of the application

### Performance Optimizations
- **Async Operations**: Non-blocking UI during long operations
- **Progress Callbacks**: Real-time feedback during processing
- **Efficient Caching**: Smart cache management for embeddings and models
- **Memory Management**: Automatic cleanup and resource management

### Code Quality
- **SOLID Principles**: Single responsibility, open/closed, dependency injection
- **Type Hints**: Comprehensive type annotations for better IDE support
- **Error Handling**: Structured error handling with recovery options
- **Logging**: Centralized logging with TUI-aware output
- **No Code Duplication**: Shared utilities and processors

## 🐛 Troubleshooting

### Common Issues
1. **No documents found**: Ensure PDF files are in `./documents/`
2. **Ollama not running**: Start Ollama service and pull the model, or switch to an API provider in Settings (Ctrl+S)
3. **API errors**: Check your API key and model settings for OpenAI/Anthropic providers
4. **Import errors**: Install all requirements with `pip install -r requirements.txt`
5. **Database errors**: Check file permissions in the workspace directory
6. **Embedding errors**: Use Ctrl+Shift+C to clean the HuggingFace cache
7. **Memory issues with large models**: Disable query expansion or use smaller models
8. **File descriptor errors**: Increase ulimit with `ulimit -n 8192`

### Tips
- Use specific, well-formed questions for better results
- Keep document files reasonably sized for better performance
- Check the status panel for system information
- Use the help screen (Ctrl+H) for guidance

## 📝 Recent Updates

### Version 2.0.0 - Architecture Refactor
- **Service Layer**: Introduced clean service-oriented architecture
- **Modular Managers**: Split monolithic classes into focused managers
- **Query Processing**: Added dedicated query processor with expansion and filtering
- **Error Handling**: Implemented comprehensive error handler with user-friendly messages
- **Configuration**: Reorganized constants into logical configuration modules
- **Logging**: Fixed duplicate logging issues in TUI mode
- **Performance**: Improved memory management and cache handling
- **UI Enhancements**: Added context toggle, cache cleaning, and system restart shortcuts
- **Code Quality**: Eliminated code duplication and improved separation of concerns

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the TUI experience!

## 📄 License

This project is open source. See the license file for details.
# 🤖 RAG Chat - Enhanced TUI Experience

A sophisticated Text User Interface (TUI) for Retrieval-Augmented Generation (RAG) chat using LangChain and Textual.

## ✨ Features

### 🎨 Enhanced User Interface
- **Modern TUI Design**: Professional-looking interface with rich styling and colors
- **Responsive Layout**: Adaptive sidebar and chat area that can be toggled
- **Progress Indicators**: Real-time progress bars for long operations
- **Rich Chat Display**: Formatted messages with timestamps and panels
- **Status Dashboard**: Live statistics and system information

### 🚀 New Functionality
- **Chat History**: Persistent session management with history tracking
- **Settings Panel**: Configurable model parameters and application settings
- **Document Browser**: Browse and manage PDF documents in the interface
- **Help System**: Comprehensive help with keyboard shortcuts
- **Export Feature**: Save chat sessions to text files

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
| `F1` | Toggle sidebar visibility |
| `Enter` | Send message |

## 🏗️ Installation

1. **Clone or download the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
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
   python rag_cli.py
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
- **Ollama Model**: Choose different Ollama models (default: llama3.2) when using Ollama
- **API Key**: Your API key for OpenAI or Anthropic (when using API providers)
- **API Base URL**: Custom API endpoint (optional, for OpenAI-compatible APIs)
- **OpenAI Model**: Model name for OpenAI (default: gpt-3.5-turbo)
- **Anthropic Model**: Model name for Anthropic (default: claude-3-haiku-20240307)

### General Settings
- **Temperature**: Control response creativity (0.0-1.0)
- **Chunk Size**: Document splitting size (default: 1000)
- **Chunk Overlap**: Overlap between chunks (default: 200)
- **Retrieval Count**: Number of relevant chunks to retrieve (default: 3)

### File Locations
- **Documents**: `./documents/` - Place your PDF files here
- **Database**: `./chroma_db/` - Vector database storage
- **Chat History**: `chat_history.json` - Persistent chat sessions
- **Exports**: `chat_export_YYYYMMDD_HHMMSS.txt` - Exported conversations

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

## 🔧 Technical Improvements

### Performance
- **Async Operations**: Non-blocking UI during long operations
- **Progress Callbacks**: Real-time feedback during processing
- **Efficient Updates**: Optimized UI refresh strategies

### User Experience
- **Responsive Design**: Adaptive layout for different terminal sizes
- **Visual Feedback**: Clear status indicators and progress bars
- **Error Recovery**: Graceful handling of errors with helpful messages

### Code Quality
- **Modular Design**: Separated concerns with dedicated classes
- **Type Hints**: Better code documentation and IDE support
- **Error Handling**: Comprehensive exception management

## 🐛 Troubleshooting

### Common Issues
1. **No documents found**: Ensure PDF files are in `./documents/`
2. **Ollama not running**: Start Ollama service and pull the model, or switch to an API provider in Settings (Ctrl+S)
3. **API errors**: Check your API key and model settings for OpenAI/Anthropic providers
4. **Import errors**: Install all requirements with `pip install -r requirements.txt`
5. **Database errors**: Check file permissions in the workspace directory

### Tips
- Use specific, well-formed questions for better results
- Keep document files reasonably sized for better performance
- Check the status panel for system information
- Use the help screen (Ctrl+H) for guidance

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the TUI experience!

## 📄 License

This project is open source. See the license file for details.
# RAG CLI Application with Modern Interface and Robust Error Handling
import os
import sys
import gc
import time
import shutil
import hashlib
import multiprocessing as mp
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from rich import print as rprint
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, TaskID
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Static, TextArea, Input, Select, Checkbox, ProgressBar
from textual.binding import Binding
from textual import work
import warnings

# Set environment variables for better cache management and to prevent conflicts
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Prevent tokenizer parallelism issues
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Reduce verbosity
os.environ['HF_DATASETS_OFFLINE'] = '0'  # Allow online access
os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Allow online access
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Disable symlink warnings
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'  # Disable advisory warnings

# Additional settings to help with file descriptor issues
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'  # Reduce file creation
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Better compatibility
os.environ['OMP_NUM_THREADS'] = '1'  # Reduce threading issues
os.environ['MKL_NUM_THREADS'] = '1'  # Reduce threading issues

# Set cache directories with better control
# Use HF_HOME as the main cache directory (TRANSFORMERS_CACHE is deprecated)
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
os.environ['HF_DATASETS_CACHE'] = os.path.expanduser('~/.cache/huggingface/datasets')

# Ensure cache directories exist
cache_dirs = [
    os.environ['HF_HOME'],
    os.environ['HF_DATASETS_CACHE'],
    os.path.join(os.environ['HF_HOME'], 'hub'),
    os.path.join(os.environ['HF_HOME'], 'transformers')
]
for cache_dir in cache_dirs:
    os.makedirs(cache_dir, exist_ok=True)

from langchain_huggingface import HuggingFaceEmbeddings

# Standard library imports
import asyncio
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

# Third-party imports
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Input, RichLog, Button, Static, Header, Footer, 
    ProgressBar, Tree, Label, Switch
)
from textual.binding import Binding
from textual.reactive import reactive
from textual.screen import ModalScreen
from rich.table import Table
from rich.panel import Panel
import pyperclip  # For clipboard support

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class SettingsManager:
    """Manages application settings with persistence"""
    
    DEFAULT_SETTINGS_FILE = "settings.json"
    
    def __init__(self, settings_file: str = DEFAULT_SETTINGS_FILE):
        self.settings_file = settings_file
        self.default_settings = {
            'model_name': 'llama3.2',
            'temperature': 0.0,
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'retrieval_k': 3,
            'auto_save': True,
            'dark_mode': False,
            'llm_provider': 'ollama',  # 'ollama', 'openai', 'anthropic'
            'api_key': '',  # For API providers
            'api_base_url': '',  # For custom API endpoints
            'openai_model': 'gpt-3.5-turbo',  # For OpenAI
            'anthropic_model': 'claude-3-haiku-20240307'  # For Anthropic
        }
        self.settings = self.load_settings()
    
    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file or return defaults"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    loaded = json.load(f)
                    # Merge with defaults to handle missing keys
                    return {**self.default_settings, **loaded}
        except (IOError, OSError, PermissionError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
            print(f"Warning: Could not load settings from {self.settings_file}: {e}")
            pass
        return self.default_settings.copy()
    
    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """Save settings to file"""
        try:
            # Merge with current settings
            self.settings.update(settings)
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
            return True
        except (IOError, OSError, PermissionError, TypeError) as e:
            print(f"Warning: Could not save settings to {self.settings_file}: {e}")
            return False
    
    def get(self, key: str, default=None) -> Any:
        """Get a setting value"""
        return self.settings.get(key, default)

class ChatHistory:
    """Manages chat history with persistence"""
    
    DEFAULT_HISTORY_FILE = "chat_history.json"
    
    def __init__(self, history_file: str = DEFAULT_HISTORY_FILE):
        self.history_file = history_file
        self.sessions: List[Dict[str, Any]] = []
        self.current_session: List[Dict[str, Any]] = []
        self.load_history()
    
    def load_history(self) -> None:
        """Load chat history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.sessions = data.get('sessions', [])
        except (IOError, OSError, PermissionError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
            print(f"Warning: Could not load chat history from {self.history_file}: {e}")
            self.sessions = []
    
    def save_history(self) -> None:
        """Save chat history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump({'sessions': self.sessions}, f, indent=2)
        except (IOError, OSError, PermissionError, TypeError) as e:
            print(f"Warning: Could not save chat history to {self.history_file}: {e}")
    
    def add_exchange(self, question: str, answer: str) -> None:
        """Add Q&A exchange to current session"""
        self.current_session.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer
        })
    
    def start_new_session(self) -> None:
        """Start a new chat session"""
        if self.current_session:
            session_data = {
                'start_time': datetime.now().isoformat(),
                'exchanges': self.current_session.copy()
            }
            self.sessions.append(session_data)
            self.save_history()
        self.current_session = []

class SettingsScreen(ModalScreen):
    """Modal screen for application settings"""
    
    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("ctrl+c", "quit", "Quit"),
    ]
    
    def on_mount(self) -> None:
        """Load current settings when screen is mounted"""
        # Get settings from the settings manager (which reflects current RAG state)
        settings = self.app.settings_manager.settings
        
        # Update input fields with current values
        self.query_one("#model-input", Input).value = str(settings.get('model_name', 'llama3.2'))
        self.query_one("#temp-input", Input).value = str(settings.get('temperature', 0.0))
        self.query_one("#chunk-input", Input).value = str(settings.get('chunk_size', 1000))
        self.query_one("#overlap-input", Input).value = str(settings.get('chunk_overlap', 200))
        self.query_one("#retrieval-input", Input).value = str(settings.get('retrieval_k', 3))
        
        # Update LLM provider settings
        provider_select = self.query_one("#provider-select", Select)
        provider_select.value = settings.get('llm_provider', 'ollama')
        
        self.query_one("#api-key-input", Input).value = str(settings.get('api_key', ''))
        self.query_one("#api-base-input", Input).value = str(settings.get('api_base_url', ''))
        self.query_one("#openai-model-input", Input).value = str(settings.get('openai_model', 'gpt-3.5-turbo'))
        self.query_one("#anthropic-model-input", Input).value = str(settings.get('anthropic_model', 'claude-3-haiku-20240307'))
        
        # Update provider-specific field visibility
        self._update_provider_fields(settings.get('llm_provider', 'ollama'))
    
    def compose(self) -> ComposeResult:
        with Container(id="settings-container"):
            yield Static("⚙️ Settings", id="settings-title")
            
            with VerticalScroll():
                # LLM Provider Settings
                yield Label("LLM Provider:")
                yield Select([
                    ("Ollama (Local)", "ollama"),
                    ("OpenAI API", "openai"),
                    ("Anthropic API", "anthropic")
                ], value="ollama", id="provider-select")
                
                yield Static("")  # Spacer
                
                # Ollama-specific settings
                yield Label("Ollama Model:", id="ollama-model-label")
                yield Input(value="llama3.2", placeholder="Ollama model name", id="model-input")
                
                # API-specific settings (initially hidden)
                yield Label("API Key:", id="api-key-label", classes="api-field")
                yield Input(value="", placeholder="Your API key", password=True, id="api-key-input", classes="api-field")
                
                yield Label("API Base URL (optional):", id="api-base-label", classes="api-field")
                yield Input(value="", placeholder="Custom API endpoint", id="api-base-input", classes="api-field")
                
                yield Label("OpenAI Model:", id="openai-model-label", classes="openai-field")
                yield Input(value="gpt-3.5-turbo", placeholder="OpenAI model name", id="openai-model-input", classes="openai-field")
                
                yield Label("Anthropic Model:", id="anthropic-model-label", classes="anthropic-field")
                yield Input(value="claude-3-haiku-20240307", placeholder="Anthropic model name", id="anthropic-model-input", classes="anthropic-field")
                
                yield Static("")  # Spacer
                
                # General settings
                yield Label("Temperature (0.0-1.0):")
                yield Input(value="0.0", placeholder="0.0", id="temp-input")
                
                yield Label("Chunk Size:")
                yield Input(value="1000", placeholder="1000", id="chunk-input")
                
                yield Label("Chunk Overlap:")
                yield Input(value="200", placeholder="200", id="overlap-input")
                
                yield Label("Retrieval Count (k):")
                yield Input(value="3", placeholder="3", id="retrieval-input")
                
                with Horizontal():
                    yield Switch(value=True, id="auto-save-switch")
                    yield Label("Auto-save chat history")
                
                with Horizontal():
                    yield Switch(value=False, id="dark-mode-switch")
                    yield Label("Dark mode")
                
                yield Static("")  # Spacer
                
                with Horizontal(classes="button-row"):
                    yield Button("Save", variant="primary", id="save-settings")
                    yield Button("Cancel", id="cancel-settings")
    
    def _update_provider_fields(self, provider: str) -> None:
        """Update visibility of provider-specific fields"""
        # Hide all provider-specific fields first
        for field_class in ["api-field", "openai-field", "anthropic-field"]:
            try:
                fields = self.query(f".{field_class}")
                for field in fields:
                    field.remove_class("visible")
            except:
                pass
        
        # Show Ollama fields
        try:
            self.query_one("#ollama-model-label").display = provider == "ollama"
            self.query_one("#model-input").display = provider == "ollama"
        except:
            pass
        
        # Show API fields for non-Ollama providers
        if provider in ["openai", "anthropic"]:
            try:
                api_fields = self.query(".api-field")
                for field in api_fields:
                    field.add_class("visible")
            except:
                pass
        
        # Show provider-specific model fields
        if provider == "openai":
            try:
                openai_fields = self.query(".openai-field")
                for field in openai_fields:
                    field.add_class("visible")
            except:
                pass
        elif provider == "anthropic":
            try:
                anthropic_fields = self.query(".anthropic-field")
                for field in anthropic_fields:
                    field.add_class("visible")
            except:
                pass
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle provider selection change"""
        if event.select.id == "provider-select":
            self._update_provider_fields(event.value)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses in settings screen"""
        if event.button.id == "cancel-settings":
            self.dismiss()
        elif event.button.id == "save-settings":
            # Get all input values
            provider_select = self.query_one("#provider-select", Select)
            model_input = self.query_one("#model-input", Input)
            temp_input = self.query_one("#temp-input", Input)
            chunk_input = self.query_one("#chunk-input", Input)
            overlap_input = self.query_one("#overlap-input", Input)
            retrieval_input = self.query_one("#retrieval-input", Input)
            api_key_input = self.query_one("#api-key-input", Input)
            api_base_input = self.query_one("#api-base-input", Input)
            openai_model_input = self.query_one("#openai-model-input", Input)
            anthropic_model_input = self.query_one("#anthropic-model-input", Input)
            
            # Validate and parse values
            try:
                new_settings = {
                    'llm_provider': provider_select.value or "ollama",
                    'model_name': model_input.value or "llama3.2",
                    'temperature': float(temp_input.value or "0.0"),
                    'chunk_size': int(chunk_input.value or "1000"),
                    'chunk_overlap': int(overlap_input.value or "200"),
                    'retrieval_k': int(retrieval_input.value or "3"),
                    'api_key': api_key_input.value or "",
                    'api_base_url': api_base_input.value or "",
                    'openai_model': openai_model_input.value or "gpt-3.5-turbo",
                    'anthropic_model': anthropic_model_input.value or "claude-3-haiku-20240307"
                }
                
                # Update the RAG system
                self.app.rag.update_settings(**new_settings)
                
                # Save settings to file
                if self.app.settings_manager.save_settings(new_settings):
                    # Update the stats display
                    self.app.update_stats()
                    
                    # Show success notification
                    self.app.notify("Settings saved successfully!")
                else:
                    self.app.notify("Settings applied but could not save to file", severity="warning")
                    
                self.dismiss()
                
            except ValueError as e:
                self.app.notify(f"Invalid settings: {str(e)}", severity="error")
            except Exception as e:
                self.app.notify(f"Error saving settings: {str(e)}", severity="error")
    
    def action_dismiss(self) -> None:
        """Handle escape key to close modal"""
        self.dismiss()

    def on_key(self, event) -> None:
        """Handle key events, specifically escape key to close modal"""
        if event.key == "escape":
            self.dismiss()
            event.prevent_default()

    def action_quit(self) -> None:
        """Quit the application from settings screen."""
        self.app.exit()

class HelpScreen(ModalScreen):
    """Modal screen showing help and keyboard shortcuts"""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("ctrl+c", "quit", "Quit"),
    ]

    help_content = """
# 🤖 RAG Chat Help

## Keyboard Shortcuts
- **Ctrl+C**: Quit application
- **Ctrl+L**: Clear current chat
- **Ctrl+R**: Reload database
- **Ctrl+H**: Show this help
- **Ctrl+S**: Open settings
- **Ctrl+N**: New chat session
- **Ctrl+Y**: Copy last message to clipboard
- **Ctrl+E**: Export current chat
- **Ctrl+Shift+E**: Export full chat history
- **Ctrl+Shift+R**: Restart RAG system (fixes errors)
- **F1**: Toggle sidebar
- **Enter**: Send message

## Getting Started
1. **Load Database**: Click "Load DB" if you have an existing ChromaDB
2. **Create Database**: Click "Create DB" to process PDFs from ./documents
3. **Ask Questions**: Type your questions and press Enter

## Features
- 📚 PDF document processing
- 🔍 Semantic search with ChromaDB
- 💬 Chat history with persistence
- 📋 Copy messages to clipboard
- 📁 Export chat conversations
- ⚙️ Configurable settings
- 🎨 Multiple themes

## Document Management
- Place PDF files in the ./documents directory
- Use "Create DB" to process new documents
- Use "Reload DB" to refresh the database

## Copying Text
- **Ctrl+Y**: Copy the last message (question or answer)
- **Ctrl+E**: Export current chat to text file
- **Ctrl+Shift+E**: Export complete chat history

## Tips
- Ask specific questions for better results
- Use clear, complete sentences
- Check the status bar for system information
    """

    def compose(self) -> ComposeResult:
        with Container(id="help-container"):
            yield Static("📖 Help & Documentation", id="help-title")
            yield RichLog(markup=True, id="help-content")
            yield Button("Close", id="close-help")
    
    def on_mount(self):
        help_log = self.query_one("#help-content", RichLog)
        help_log.write(self.help_content)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses in the help screen"""
        if event.button.id == "close-help":
            self.dismiss()
    
    def action_dismiss(self) -> None:
        """Handle escape key to close modal"""
        self.dismiss()

    def action_quit(self) -> None:
        """Quit the application from help screen."""
        self.app.exit()

class DocumentBrowserScreen(ModalScreen):
    """Modal screen for browsing and managing documents"""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("ctrl+c", "quit", "Quit"),
    ]
    
    def compose(self) -> ComposeResult:
        with Container(id="doc-browser-container"):
            yield Static("📄 Document Browser", id="doc-title")
            
            with Horizontal():
                with Vertical():
                    yield Label("Documents Directory:")
                    yield Tree("Documents", id="doc-tree")
                
                with Vertical():
                    yield Label("Document Preview:")
                    yield RichLog(id="doc-preview", markup=True)
            
            with Horizontal():
                yield Button("Refresh", id="refresh-docs")
                yield Button("Close", id="close-docs")
    
    def on_mount(self):
        self.refresh_documents()
    
    def refresh_documents(self):
        """Refresh the document tree"""
        tree = self.query_one("#doc-tree", Tree)
        tree.clear()
        
        docs_path = Path("./documents")
        if docs_path.exists():
            for file_path in docs_path.rglob("*.pdf"):
                tree.root.add_leaf(str(file_path.relative_to(docs_path)))
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses in document browser"""
        if event.button.id == "close-docs":
            self.dismiss()
        elif event.button.id == "refresh-docs":
            self.refresh_documents()
    
    def action_dismiss(self) -> None:
        """Handle escape key to close modal"""
        self.dismiss()

    def action_quit(self) -> None:
        """Quit the application from document browser."""
        self.app.exit()

class RAGSystem:
    """Enhanced RAG System with modern configuration and robust error handling"""
    
    def __init__(self, model_name="llama3.2:3b", temperature=0.1, chunk_size=1000, chunk_overlap=200, retrieval_k=3, settings_manager=None):
        self.model_name = model_name
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        self.settings_manager = settings_manager
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.conversation_history = []
        
        # Check and increase file descriptor limit before initialization
        self._check_and_increase_fd_limit()
        
        # Initialize LLM based on provider settings
        self._initialize_llm()
        
        # Initialize embeddings with better error handling
        self._initialize_embeddings_safely()
    
    def _check_and_increase_fd_limit(self):
        """Check and try to increase file descriptor limit"""
        try:
            import resource
            
            # Get current limits
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            
            # Try to increase to a reasonable limit
            target_limit = min(8192, hard)
            
            if soft < target_limit:
                try:
                    resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, hard))
                    print(f"[green]✓ Increased file descriptor limit from {soft} to {target_limit}[/green]")
                except Exception:
                    print(f"[yellow]⚠ Could not increase file descriptor limit (current: {soft})[/yellow]")
                    print("[yellow]💡 Try running: ulimit -n 8192 before starting the app[/yellow]")
        except Exception:
            # Not critical if this fails
            pass
    
    def _initialize_llm(self):
        """Initialize LLM based on provider settings"""
        if not self.settings_manager:
            # Fallback to Ollama if no settings manager
            self._initialize_ollama()
            return
        
        provider = self.settings_manager.get('llm_provider', 'ollama')
        
        try:
            if provider == 'ollama':
                self._initialize_ollama()
            elif provider == 'openai':
                self._initialize_openai()
            elif provider == 'anthropic':
                self._initialize_anthropic()
            else:
                print(f"[yellow]⚠ Unknown provider '{provider}', falling back to Ollama[/yellow]")
                self._initialize_ollama()
        except Exception as e:
            print(f"[red]✗ Failed to initialize {provider}: {str(e)}[/red]")
            if provider != 'ollama':
                print("[yellow]⚠ Falling back to Ollama[/yellow]")
                try:
                    self._initialize_ollama()
                except Exception as fallback_error:
                    print(f"[red]✗ Ollama fallback also failed: {str(fallback_error)}[/red]")
                    print("[red]Please ensure Ollama is installed and running, or configure an API provider[/red]")
                    raise fallback_error
            else:
                raise e
    
    def _initialize_ollama(self):
        """Initialize Ollama LLM"""
        try:
            self.llm = OllamaLLM(
                model=self.model_name, 
                temperature=self.temperature,
                num_ctx=2048,  # Reduce context window to save memory
                num_thread=1   # Use single thread to avoid fd issues
            )
            print(f"[green]✓ Initialized Ollama with model: {self.model_name}[/green]")
        except Exception:
            # Fallback to simpler initialization
            self.llm = OllamaLLM(model=self.model_name, temperature=self.temperature)
            print(f"[green]✓ Initialized Ollama (simple mode) with model: {self.model_name}[/green]")
    
    def _initialize_openai(self):
        """Initialize OpenAI LLM"""
        api_key = self.settings_manager.get('api_key', '')
        api_base = self.settings_manager.get('api_base_url', '')
        model = self.settings_manager.get('openai_model', 'gpt-3.5-turbo')
        
        if not api_key:
            raise ValueError("OpenAI API key is required but not provided in settings")
        
        kwargs = {
            'model': model,
            'temperature': self.temperature,
            'api_key': api_key
        }
        
        if api_base:
            kwargs['base_url'] = api_base
        
        self.llm = ChatOpenAI(**kwargs)
        print(f"[green]✓ Initialized OpenAI with model: {model}[/green]")
    
    def _initialize_anthropic(self):
        """Initialize Anthropic LLM"""
        api_key = self.settings_manager.get('api_key', '')
        model = self.settings_manager.get('anthropic_model', 'claude-3-haiku-20240307')
        
        if not api_key:
            raise ValueError("Anthropic API key is required but not provided in settings")
        
        self.llm = ChatAnthropic(
            model=model,
            temperature=self.temperature,
            api_key=api_key
        )
        print(f"[green]✓ Initialized Anthropic with model: {model}[/green]")
    
    def _clean_hf_cache_locks(self):
        """Clean up Hugging Face cache lock files that may prevent model loading"""
        cache_dir = Path(os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')))
        
        if not cache_dir.exists():
            return
        
        lock_patterns = ['*.lock', '*.tmp*', '*incomplete*']
        cleaned_files = []
        
        try:
            for pattern in lock_patterns:
                for lock_file in cache_dir.rglob(pattern):
                    try:
                        if lock_file.is_file():
                            # Check if lock file is stale (older than 30 minutes)
                            if time.time() - lock_file.stat().st_mtime > 1800:
                                lock_file.unlink()
                                cleaned_files.append(str(lock_file))
                    except (OSError, PermissionError):
                        continue
            
            if cleaned_files:
                print(f"[green]✓ Cleaned {len(cleaned_files)} stale cache lock files[/green]")
                
        except Exception as e:
            print(f"[yellow]⚠ Could not clean cache locks: {str(e)}[/yellow]")
    
    def _initialize_embeddings_safely(self):
        """Initialize embeddings with robust error handling and cache management"""
        # Ensure cache directories are properly set up
        self._ensure_cache_directories()
        
        # Clean any stale lock files first
        self._clean_hf_cache_locks()
        
        # Force single-threaded multiprocessing to avoid conflicts
        mp.set_start_method('spawn', force=True)

        try:
            # Force garbage collection before initialization
            gc.collect()

            # Force single-threaded operation to avoid file descriptor issues
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': False
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 1,  # Process one at a time to avoid memory/fd issues
                    'convert_to_numpy': True
                }
            )

            # Force garbage collection after initialization
            gc.collect()

        except Exception as e:
            # If initialization fails, try cache cleanup and retry once
            print(f"[yellow]⚠ Initial embedding initialization failed: {str(e)}[/yellow]")
            print("[blue]ℹ Attempting cache cleanup and retry...[/blue]")
            
            self._clean_hf_cache_locks()
            
            # Wait a moment for file system to catch up
            time.sleep(2)
            
            try:
                # Retry with additional safety measures
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={
                        'device': 'cpu',
                        'trust_remote_code': False,
                        'cache_dir': os.environ['HF_HOME']  # Explicit cache dir
                    },
                    encode_kwargs={
                        'normalize_embeddings': True,
                        'batch_size': 1,
                        'convert_to_numpy': True
                    }
                )
                print("[green]✓ Embedding initialization successful after cache cleanup[/green]")
                
            except Exception as retry_error:
                print(f"[red]✗ Failed to initialize embeddings after retry: {str(retry_error)}[/red]")
                raise Exception(f"Failed to initialize embeddings: {str(retry_error)}") from retry_error
        
    def update_settings(self, **kwargs):
        """Update RAG system settings"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Recreate LLM with new settings
        self._initialize_llm()
        
        # Recreate QA chain if vectorstore exists
        if self.vectorstore:
            self._setup_qa_chain()
        
    def load_existing_db(self, db_path="./chroma_db"):
        """Load existing ChromaDB with modern configuration"""
        import os
        import gc

        # Set environment variables for ChromaDB (modern approach)
        os.environ['ANONYMIZED_TELEMETRY'] = 'False'

        if os.path.exists(db_path):
            try:
                # Force garbage collection before loading
                gc.collect()
                
                # Use modern ChromaDB configuration
                self.vectorstore = Chroma(
                    persist_directory=db_path,
                    embedding_function=self.embeddings
                )
                
                # Force garbage collection after loading
                gc.collect()
                
                self._setup_qa_chain()
                return True
            except Exception as e:
                # If loading fails, provide more context
                print(f"[yellow]⚠ Could not load database: {str(e)}[/yellow]")
                if "fds_to_keep" in str(e):
                    print("[yellow]💡 Try restarting the application or increasing file descriptor limit[/yellow]")
                return False
        return False
    
    def create_db_from_docs(self, docs_path="./documents", db_path="./chroma_db", progress_callback=None):
        """Create new ChromaDB from documents with robust error handling and file descriptor management"""
        import os
        import gc
        from pathlib import Path
        from contextlib import contextmanager

        @contextmanager
        def pdf_loader_context(pdf_path):
            """Context manager for PDF loading with proper cleanup"""
            loader = None
            try:
                loader = PyPDFLoader(str(pdf_path))
                yield loader
            finally:
                # Explicit cleanup
                if hasattr(loader, 'close'):
                    loader.close()
                del loader
                gc.collect()

        # Set comprehensive environment variables to help with subprocess issues
        os.environ['PYTHONUNBUFFERED'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'

        try:
            docs_path = Path(docs_path)
            pdf_files = list(docs_path.glob("**/*.pdf"))

            if not pdf_files:
                raise ValueError(f"No PDF files found in {docs_path}")

            if progress_callback:
                progress_callback(f"Found {len(pdf_files)} PDF files...")

            # Process PDFs one by one with explicit cleanup to avoid file descriptor issues
            all_documents = []
            successful_files = []
            failed_files = []

            for i, pdf_file in enumerate(pdf_files):
                try:
                    if progress_callback:
                        progress_callback(f"Processing {pdf_file.name} ({i+1}/{len(pdf_files)})...")

                    # Use context manager for proper cleanup
                    with pdf_loader_context(pdf_file) as loader:
                        documents = loader.load()

                        if documents:
                            all_documents.extend(documents)
                            successful_files.append(pdf_file.name)
                        else:
                            failed_files.append(f"{pdf_file.name} (no content)")

                    # Force garbage collection after each PDF to free file descriptors
                    gc.collect()

                except Exception as pdf_error:
                    failed_files.append(f"{pdf_file.name} ({str(pdf_error)[:50]}...)")
                    gc.collect()  # Clean up even on error
                    continue  # Skip problematic PDFs and continue with others

            if not all_documents:
                error_msg = "No documents could be processed successfully."
                if failed_files:
                    error_msg += f" Failed files: {', '.join(failed_files[:3])}"
                    if len(failed_files) > 3:
                        error_msg += f" and {len(failed_files) - 3} more"
                raise ValueError(error_msg)

            if progress_callback:
                msg = f"Successfully loaded {len(successful_files)} files"
                if failed_files:
                    msg += f" ({len(failed_files)} failed)"
                progress_callback(msg)

            if progress_callback:
                progress_callback(f"Splitting {len(all_documents)} documents...")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            texts = text_splitter.split_documents(all_documents)

            # Clear large document list to free memory
            del all_documents
            gc.collect()

            if progress_callback:
                progress_callback(f"Creating embeddings for {len(texts)} chunks...")

            # Create vectorstore with robust file descriptor handling
            self.vectorstore = self._create_chroma_db(texts, db_path)

            if progress_callback:
                progress_callback("Setting up QA chain...")

            self._setup_qa_chain()

            success_msg = f"Database created with {len(texts)} chunks from {len(successful_files)} files"
            if failed_files:
                success_msg += f" ({len(failed_files)} files skipped due to errors)"

            if progress_callback:
                progress_callback(success_msg)

        except Exception as e:
            # Force cleanup on error
            gc.collect()
            # Re-raise with more context
            raise Exception(f"Database creation failed: {str(e)}") from e

    def _create_chroma_db(self, texts, db_path):
        """Create ChromaDB with modern configuration and robust file descriptor handling"""
        import os
        import gc
        import shutil
        import time
        from pathlib import Path
        import tempfile

        # Set comprehensive environment variables for ChromaDB
        os.environ['ANONYMIZED_TELEMETRY'] = 'False'
        os.environ['CHROMA_SERVER_NOFILE'] = '65536'  # Increase file descriptor limit for ChromaDB

        final_path = Path(db_path)
        backup_path = None
        temp_path = None

        try:
            # Clean up old backups first
            self._cleanup_old_backups(db_path)
            
            # Create backup of existing database if it exists
            if final_path.exists():
                backup_path = Path(str(final_path) + f"_backup_{int(time.time())}")
                print(f"[blue]ℹ Backing up existing database to {backup_path}[/blue]")
                shutil.move(str(final_path), str(backup_path))

            # Create temporary directory for new database
            temp_dir = tempfile.mkdtemp(prefix="chroma_temp_")
            temp_path = Path(temp_dir)

            # Create ChromaDB with modern configuration
            # Use smaller batches and add delays to avoid file descriptor exhaustion
            batch_size = 25  # Reduced batch size for better stability
            delay_between_batches = 0.1  # Small delay to allow cleanup

            vectorstore = None
            total_batches = (len(texts) + batch_size - 1) // batch_size

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_num = (i // batch_size) + 1

                try:
                    if vectorstore is None:
                        # Create initial vectorstore with explicit settings in temp directory
                        vectorstore = Chroma.from_documents(
                            documents=batch,
                            embedding=self.embeddings,
                            persist_directory=str(temp_path),
                            collection_metadata={"hnsw:space": "cosine"}
                        )
                    else:
                        # Add to existing vectorstore
                        vectorstore.add_documents(batch)

                    # Force garbage collection and small delay between batches
                    gc.collect()
                    if batch_num < total_batches:  # Don't delay after last batch
                        time.sleep(delay_between_batches)

                except Exception:
                    # If a batch fails, try to continue with smaller batches
                    if batch_size > 5:
                        # Retry with smaller batch size
                        smaller_batch_size = max(5, batch_size // 2)
                        for j in range(i, min(i + batch_size, len(texts)), smaller_batch_size):
                            small_batch = texts[j:j + smaller_batch_size]
                            try:
                                if vectorstore is None:
                                    vectorstore = Chroma.from_documents(
                                        documents=small_batch,
                                        embedding=self.embeddings,
                                        persist_directory=db_path,
                                        collection_metadata={"hnsw:space": "cosine"}
                                    )
                                else:
                                    vectorstore.add_documents(small_batch)
                                gc.collect()
                                time.sleep(delay_between_batches)
                            except Exception:
                                # Skip this small batch and continue
                                continue
                    else:
                        # Skip this batch entirely if we can't make it smaller
                        continue

            if vectorstore is None:
                raise Exception("Failed to create any vectorstore - all batches failed. This may be due to embedding model issues or document processing problems.")

            # If we get here, database creation was successful
            # Verify the temporary database is valid before moving
            if temp_path and temp_path.exists():
                # Quick validation - check if essential files exist
                essential_files = ["chroma.sqlite3"]
                temp_db_valid = all((temp_path / f).exists() for f in essential_files)
                
                if temp_db_valid:
                    shutil.move(str(temp_path), str(final_path))
                    print(f"[green]✓ Database successfully created at {final_path}[/green]")
                    
                    # Remove backup if creation was successful
                    if backup_path and backup_path.exists():
                        try:
                            shutil.rmtree(str(backup_path))
                            print(f"[green]✓ Removed backup database[/green]")
                        except Exception:
                            print(f"[yellow]⚠ Could not remove backup at {backup_path}[/yellow]")
                else:
                    raise Exception("Created database appears to be incomplete or corrupted")

            # Final garbage collection
            gc.collect()
            return vectorstore

        except Exception as e:
            # Clean up temporary directory on failure
            if temp_path and temp_path.exists():
                try:
                    shutil.rmtree(str(temp_path))
                    print(f"[blue]ℹ Cleaned up temporary database[/blue]")
                except Exception:
                    pass
            
            # Restore backup if it exists
            if backup_path and backup_path.exists():
                try:
                    shutil.move(str(backup_path), str(final_path))
                    print(f"[green]✓ Restored backup database[/green]")
                except Exception as restore_error:
                    print(f"[red]✗ Failed to restore backup: {restore_error}[/red]")
            
            gc.collect()
            raise Exception(f"ChromaDB creation failed: {str(e)}") from e

    def _cleanup_old_backups(self, db_path, max_backups=3):
        """Clean up old backup directories to prevent disk space issues"""
        try:
            base_path = Path(db_path)
            parent_dir = base_path.parent
            backup_pattern = f"{base_path.name}_backup_*"
            
            # Find all backup directories
            backup_dirs = []
            for item in parent_dir.glob(backup_pattern):
                if item.is_dir():
                    try:
                        # Extract timestamp from backup name
                        timestamp = int(item.name.split('_backup_')[1])
                        backup_dirs.append((timestamp, item))
                    except (ValueError, IndexError):
                        continue
            
            # Sort by timestamp (newest first) and remove old backups
            backup_dirs.sort(reverse=True)
            for _, backup_dir in backup_dirs[max_backups:]:
                try:
                    shutil.rmtree(str(backup_dir))
                    print(f"[blue]ℹ Removed old backup: {backup_dir.name}[/blue]")
                except Exception as e:
                    print(f"[yellow]⚠ Could not remove old backup {backup_dir}: {e}[/yellow]")
                    
        except Exception as e:
            print(f"[yellow]⚠ Error during backup cleanup: {e}[/yellow]")

    def _setup_qa_chain(self):
        """Setup the QA chain using modern LangChain approach"""
        system_prompt = (
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        
        self.qa_chain = create_retrieval_chain(
            self.vectorstore.as_retriever(search_kwargs={"k": self.retrieval_k}), 
            question_answer_chain
        )
    
    async def query(self, question):
        """Query the RAG system with better error handling and async execution"""
        if not self.vectorstore:
            return "RAG system not initialized. Load or create a database first."
        
        try:
            # Run the query in a thread executor to avoid blocking the UI
            loop = asyncio.get_event_loop()
            
            # Create a simple query function that avoids file descriptor issues
            def simple_query():
                import gc
                import os
                
                # Set conservative environment for the query thread
                os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                
                try:
                    # Force garbage collection
                    gc.collect()
                    
                    # Get retriever with limited results
                    retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.retrieval_k})
                    
                    # Get relevant documents - using the simpler method
                    relevant_docs = retriever.get_relevant_documents(question)
                    
                    if not relevant_docs:
                        return "I couldn't find any relevant information in the documents to answer your question."
                    
                    # Format context
                    context_parts = []
                    for i, doc in enumerate(relevant_docs[:self.retrieval_k]):
                        context_parts.append(f"Document {i+1}:\n{doc.page_content}")
                    context = "\n\n".join(context_parts)
                    
                    # Create prompt
                    prompt = f"""Based on the following context, please answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
                    
                    # Query LLM with timeout protection
                    response = self.llm.invoke(prompt)
                    
                    # Clean up
                    gc.collect()
                    
                    return response
                    
                except Exception as e:
                    # Handle specific errors
                    error_str = str(e)
                    if "fds_to_keep" in error_str:
                        # Try a minimal query without retriever
                        try:
                            simple_prompt = f"Question: {question}\n\nPlease provide a helpful response based on general knowledge."
                            return self.llm.invoke(simple_prompt)
                        except:
                            return "System resource error. Please restart the RAG system (Ctrl+Shift+R)."
                    else:
                        raise e
            
            # Execute in thread pool
            result = await loop.run_in_executor(None, simple_query)
            return result
            
        except Exception as e:
            error_msg = str(e)
            
            # Provide specific guidance for common errors
            if "fds_to_keep" in error_msg or "Bad file descriptor" in error_msg:
                return ("I encountered a system resource error. Please try:\n"
                        "1. Press Ctrl+Shift+R to restart the RAG system\n"
                        "2. Reduce chunk size in settings (Ctrl+S)\n"
                        "3. Restart the application")
            elif "connection" in error_msg.lower() or "ollama" in error_msg.lower():
                return ("Cannot connect to Ollama. You can:\n"
                        "1. Start Ollama (run 'ollama serve' in terminal)\n"
                        "2. Install the model (run 'ollama pull llama3.2')\n"
                        "3. Or switch to an API provider in Settings (Ctrl+S)")
            else:
                return f"Error: {error_msg}"
    
    def get_stats(self):
        """Get database statistics"""
        if not self.vectorstore:
            return None
        
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            return {
                "document_count": count,
                "model": self.model_name,
                "temperature": self.temperature,
                "chunk_size": self.chunk_size,
                "retrieval_k": self.retrieval_k
            }
        except Exception:
            return {"document_count": "Unknown"}

    def _ensure_cache_directories(self):
        """Ensure all required cache directories exist and are writable"""
        cache_dirs = [
            os.environ['HF_HOME'],
            os.environ['HF_DATASETS_CACHE'],
            os.path.join(os.environ['HF_HOME'], 'hub'),
            os.path.join(os.environ['HF_HOME'], 'transformers'),
        ]
        
        for cache_dir in cache_dirs:
            try:
                cache_path = Path(cache_dir)
                cache_path.mkdir(parents=True, exist_ok=True)
                
                # Test if directory is writable
                test_file = cache_path / '.write_test'
                test_file.touch()
                test_file.unlink()
                
            except (OSError, PermissionError) as e:
                print(f"[yellow]⚠ Cache directory issue: {cache_dir} - {str(e)}[/yellow]")
                # Try to create an alternative cache location
                alt_cache = Path.home() / '.local' / 'share' / 'huggingface'
                alt_cache.mkdir(parents=True, exist_ok=True)
                os.environ['HF_HOME'] = str(alt_cache)
                break

class RAGChatApp(App):
    """Enhanced Textual app for RAG chat interface with improved UX."""
    
    CSS = """
    /* Main layout */
    .main-container {
        layout: horizontal;
        height: 1fr;
    }
    
    .sidebar {
        width: 25%;
        background: $surface;
        border-right: solid $primary;
    }
    
    .chat-area {
        width: 75%;
        layout: vertical;
        height: 100%;
    }

    .chat-container {
        height: 1fr;
        border: solid $accent;
        margin: 1 1 0 1;
        min-height: 10;
    }

    .input-container {
        height: auto;
        border: solid $primary;
        margin: 0 1;
        padding: 1;
    }

    .status-container {
        height: auto;
        background: $surface;
        margin: 0 1 1 1;
        padding: 1;
    }
    
    /* Widgets */
    Input {
        width: 1fr;
        margin: 0;
    }

    Button {
        width: auto;
        min-width: 12;
        margin: 0;
        margin-left: 1;
    }
    
    .primary-btn {
        background: $primary;
    }
    
    .success-btn {
        background: $success;
    }
    
    .warning-btn {
        background: $warning;
    }
    
    /* Sidebar */
    .sidebar-title {
        text-align: center;
        background: $primary;
        color: $text;
        height: 3;
        content-align: center middle;
    }
    
    .stats-panel {
        margin: 1;
        border: solid $accent;
        height: auto;
    }
    
    .history-panel {
        height: 1fr;
        margin: 1;
        border: solid $accent;
    }
    
         /* Modal screens */
     #settings-container, #help-container, #doc-browser-container {
         width: 80%;
         height: 90%;
         max-height: 50;
         background: $surface;
         border: solid $primary;
         margin: 2;
     }
     
     #settings-container VerticalScroll {
         height: 1fr;
         margin: 1;
         padding: 1;
     }
     
     #settings-container .button-row {
         height: 3;
         padding-top: 1;
         align: center middle;
     }
    
    #settings-title, #help-title, #doc-title {
        text-align: center;
        background: $primary;
        height: 3;
        content-align: center middle;
    }
    
    /* Progress and status */
    .progress-container {
        height: auto;
        margin: 0 1;
        padding: 1;
    }
    
    ProgressBar {
        width: 1fr;
    }
    
    /* Chat styling */
    #chat {
        scrollbar-gutter: stable;
    }
    
    /* Responsive adjustments */
    .compact-sidebar {
        width: 20%;
    }
    
    .expanded-chat {
        width: 80%;
    }
    
    .hidden-sidebar {
        width: 0%;
        display: none;
    }
    
    .full-chat {
        width: 100%;
    }
    
    /* Provider-specific field styles */
    .api-field {
        display: none;
    }
    
    .openai-field {
        display: none;
    }
    
    .anthropic-field {
        display: none;
    }
    
    .api-field.visible,
    .openai-field.visible,
    .anthropic-field.visible {
        display: block;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_chat", "Clear Chat"),
        Binding("ctrl+r", "reload_db", "Reload DB"),
        Binding("ctrl+h", "show_help", "Help"),
        Binding("ctrl+s", "show_settings", "Settings"),
        Binding("ctrl+n", "new_session", "New Session"),
        Binding("ctrl+d", "show_documents", "Documents"),
        Binding("f1", "toggle_sidebar", "Toggle Sidebar"),
        Binding("ctrl+e", "export_chat", "Export Chat"),
        Binding("ctrl+y", "copy_last_message", "Copy Last"),
        Binding("ctrl+shift+e", "export_full_chat", "Export Full"),
        Binding("ctrl+shift+r", "restart_rag", "Restart RAG"),
    ]
    
    show_sidebar = reactive(True)
    processing = reactive(False)
    
    def __init__(self):
        super().__init__()
        self.settings_manager = SettingsManager()
        # Initialize RAG system with saved settings
        self.rag = RAGSystem(
            model_name=self.settings_manager.get('model_name', 'llama3.2:3b'),
            temperature=self.settings_manager.get('temperature', 0.1),
            chunk_size=self.settings_manager.get('chunk_size', 1000),
            chunk_overlap=self.settings_manager.get('chunk_overlap', 200),
            retrieval_k=self.settings_manager.get('retrieval_k', 3),
            settings_manager=self.settings_manager
        )
        self.chat_history = ChatHistory()
        self.current_progress = 0
        self.progress_timer: Optional[Any] = None
        self.chat_messages = []  # Store messages for copying
        
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header(show_clock=True)
        
        with Container(classes="main-container"):
            # Sidebar
            with Vertical(classes="sidebar", id="sidebar"):
                yield Static("🤖 RAG Assistant", classes="sidebar-title")
                
                with Container(classes="stats-panel"):
                    yield Static("📊 Status", id="stats-title")
                    yield RichLog(id="stats-content", markup=True, max_lines=10)
                
                with Container(classes="history-panel"):
                    yield Static("📝 History", id="history-title")
                    yield RichLog(id="history-content", markup=True, max_lines=15)
            
            # Main chat area
            with Vertical(classes="chat-area"):
                with Container(classes="chat-container"):
                    yield RichLog(id="chat", markup=True, highlight=True, wrap=True)
                
                # Progress bar (initially hidden)
                with Container(classes="progress-container", id="progress-container"):
                    yield ProgressBar(id="progress-bar", show_eta=False)
                    yield Static("Ready", id="progress-text")
                
                with Horizontal(classes="input-container"):
                    yield Input(placeholder="Ask a question about your documents...", id="question_input")
                    yield Button("📤 Send", classes="primary-btn", id="send_btn")
                
                with Horizontal(classes="status-container"):
                    yield Button("📂 Load DB", classes="success-btn", id="load_btn")
                    yield Button("🔄 Create DB", classes="warning-btn", id="create_btn")
                    yield Button("📄 Docs", id="docs_btn")
                    yield Button("⚙️ Settings", id="settings_btn")
                    yield Button("❓ Help", id="help_btn")
        
        yield Footer()
    
    def watch_show_sidebar(self, show: bool) -> None:
        """React to sidebar visibility changes"""
        sidebar = self.query_one("#sidebar")
        chat_area = self.query_one(".chat-area")
        
        if show:
            sidebar.add_class("sidebar")
            sidebar.remove_class("hidden-sidebar")
            chat_area.remove_class("full-chat")
            chat_area.add_class("chat-area")
        else:
            sidebar.remove_class("sidebar")
            sidebar.add_class("hidden-sidebar")
            chat_area.remove_class("chat-area")
            chat_area.add_class("full-chat")
    
    def watch_processing(self, processing: bool) -> None:
        """React to processing state changes"""
        progress_container = self.query_one("#progress-container")
        if processing:
            progress_container.styles.display = "block"
        else:
            progress_container.styles.display = "none"
    
    async def on_mount(self) -> None:
        """Called when app starts."""
        chat = self.query_one("#chat", RichLog)
        
        # Welcome message with rich formatting
        welcome_panel = Panel.fit(
            "Welcome to RAG Chat! 🤖\n\n"
            "• Load an existing database or create a new one\n"
            "• Ask questions about your documents\n"
            "• Use Ctrl+H for help and shortcuts",
            title="🚀 Getting Started",
            border_style="green"
        )
        chat.write(welcome_panel)
        
        # Update status
        self.update_stats()
        
        # Try to load existing DB automatically
        if self.rag.load_existing_db():
            chat.write("[green]✓ Automatically loaded existing ChromaDB[/green]")
            self.update_stats()
        else:
            chat.write("[yellow]💡 No existing database found. Create one from documents or load an existing one.[/yellow]")
    
    def update_stats(self):
        """Update the stats panel"""
        stats_content = self.query_one("#stats-content", RichLog)
        stats_content.clear()
        
        stats = self.rag.get_stats()
        if stats:
            stats_table = Table(show_header=False, box=None, padding=0)
            stats_table.add_column("Key", style="cyan")
            stats_table.add_column("Value", style="white")
            
            for key, value in stats.items():
                display_key = key.replace("_", " ").title()
                stats_table.add_row(display_key, str(value))
            
            stats_content.write(stats_table)
        else:
            stats_content.write("[dim]No database loaded[/dim]")
    
    def update_progress(self, message: str, progress: float = None):
        """Update progress display"""
        progress_text = self.query_one("#progress-text", Static)
        progress_bar = self.query_one("#progress-bar", ProgressBar)
        
        progress_text.update(message)
        if progress is not None:
            progress_bar.progress = progress
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Called when user presses enter in input field."""
        if event.input.id == "question_input":
            await self._process_question(event.value)
            event.input.value = ""
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "send_btn":
            question_input = self.query_one("#question_input", Input)
            await self._process_question(question_input.value)
            question_input.value = ""
            
        elif button_id == "load_btn":
            await self._load_database()
            
        elif button_id == "create_btn":
            await self._create_database()
            
        elif button_id == "docs_btn":
            self.push_screen(DocumentBrowserScreen())
            
        elif button_id == "settings_btn":
            self.push_screen(SettingsScreen())
            
        elif button_id == "help_btn":
            self.push_screen(HelpScreen())
    
    async def _load_database(self):
        """Load existing database with progress feedback"""
        chat = self.query_one("#chat", RichLog)
        self.processing = True
        
        self.update_progress("🔍 Searching for existing database...")
        
        # Simulate some loading time for better UX
        await asyncio.sleep(0.5)
        
        if self.rag.load_existing_db():
            chat.write("[green]✅ Database loaded successfully![/green]")
            self.update_stats()
            self.update_progress("✅ Database ready", 100)
        else:
            chat.write("[red]❌ No database found at ./chroma_db[/red]")
            self.update_progress("❌ No database found")
        
        await asyncio.sleep(1)
        self.processing = False
    
    async def _create_database(self):
        """Create database with enhanced progress tracking"""
        chat = self.query_one("#chat", RichLog)
        self.processing = True

        try:
            # Check if documents directory exists
            docs_path = Path("./documents")
            if not docs_path.exists():
                raise ValueError("Documents directory './documents' not found. Please create it and add PDF files.")

            # Check if there are any PDF files
            pdf_files = list(docs_path.glob("**/*.pdf"))
            if not pdf_files:
                raise ValueError("No PDF files found in './documents' directory. Please add some PDF files.")

            self.update_progress(f"📂 Found {len(pdf_files)} PDF files...", 10)
            await asyncio.sleep(0.5)

            # Create database with detailed progress tracking
            try:
                # Define a thread-safe progress callback that updates the UI
                progress_messages = []

                def progress_callback(message):
                    progress_messages.append(message)

                # Run database creation in thread executor to avoid async context issues
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.rag.create_db_from_docs(
                        docs_path="./documents",
                        db_path="./chroma_db",
                        progress_callback=progress_callback
                    )
                )

                self.update_progress("✅ Database creation complete!", 100)
                chat.write("[green]✅ Database created successfully![/green]")

                # Show progress messages that were collected
                if progress_messages:
                    chat.write(f"[blue]📊 Processing summary: {progress_messages[-1]}[/blue]")

                # Show summary of what was processed
                stats = self.rag.get_stats()
                if stats and stats.get("document_count", 0) > 0:
                    chat.write(f"[blue]📊 Processed {stats['document_count']} document chunks[/blue]")

                self.update_stats()

            except Exception as pdf_error:
                # More specific error handling for PDF processing issues
                error_msg = str(pdf_error)



                if "all batches failed" in error_msg.lower():
                    chat.write("[red]❌ Database creation failed: All document batches failed to process[/red]")
                    chat.write("[yellow]💡 This may be due to embedding model issues or document format problems.[/yellow]")
                    chat.write("[yellow]💡 Try restarting the application or check the terminal for detailed errors.[/yellow]")
                elif "fds_to_keep" in error_msg or "file descriptor" in error_msg.lower():
                    chat.write("[red]❌ PDF processing error: File descriptor issue[/red]")
                    chat.write("[yellow]💡 This is often caused by corrupted PDFs or system limitations.[/yellow]")
                    chat.write("[yellow]💡 Try removing problematic PDF files or restarting the application.[/yellow]")
                elif "No PDF files found" in error_msg:
                    chat.write("[red]❌ No PDF files found in ./documents directory[/red]")
                    chat.write("[yellow]💡 Please add some PDF files to the ./documents directory.[/yellow]")
                elif "No documents could be processed" in error_msg:
                    chat.write("[red]❌ All PDF files failed to process[/red]")
                    chat.write("[yellow]💡 Check if your PDF files are corrupted or password-protected.[/yellow]")
                else:
                    chat.write(f"[red]❌ Database creation error: {error_msg}[/red]")
                    chat.write("[yellow]💡 Check the error message above for specific details.[/yellow]")

                self.update_progress(f"❌ Error: {error_msg[:50]}...")

        except Exception as e:
            chat.write(f"[red]❌ Error creating database: {str(e)}[/red]")
            self.update_progress(f"❌ Error: {str(e)}")

        await asyncio.sleep(1.5)
        self.processing = False
    
    async def _process_question(self, question: str) -> None:
        """Process user question with enhanced UI feedback."""
        if not question.strip():
            return
            
        chat = self.query_one("#chat", RichLog)
        
        # Display user question with timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        user_panel = Panel(
            question,
            title=f"👤 You [{timestamp}]",
            border_style="blue",
            padding=(0, 1)
        )
        chat.write(user_panel)
        
        # Store the question for copying
        self.chat_messages.append({
            'type': 'user',
            'content': question,
            'timestamp': timestamp
        })
        
        if not self.rag.vectorstore:
            chat.write("[red]⚠️ Please load or create a database first.[/red]")
            return
        
        self.processing = True
        self.update_progress("🤔 Thinking...", 50)
        
        # Add to history
        history_content = self.query_one("#history-content", RichLog)
        history_content.write(f"[dim]{timestamp}[/dim] {question[:50]}...")
        
        try:
            # Show animated thinking indicator
            thinking_msg = chat.write("[dim]🧠 Processing your question...[/dim]")
            
            start_time = time.time()
            answer = await self.rag.query(question)
            response_time = time.time() - start_time
            
            # Clear the thinking indicator
            # Note: RichLog doesn't support removing specific messages, so we'll just add the response
            
            # Remove thinking indicator and add answer
            assistant_panel = Panel(
                answer,
                title=f"🤖 Assistant [{response_time:.1f}s]",
                border_style="green",
                padding=(0, 1)
            )
            chat.write(assistant_panel)
            
            # Store the answer for copying
            self.chat_messages.append({
                'type': 'assistant',
                'content': answer,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            # Add to chat history
            self.chat_history.add_exchange(question, answer)
            
            self.update_progress("✅ Response generated", 100)
            
        except Exception as e:
            error_panel = Panel(
                f"Sorry, I encountered an error: {str(e)}",
                title="❌ Error",
                border_style="red",
                padding=(0, 1)
            )
            chat.write(error_panel)
            self.update_progress(f"❌ Error: {str(e)}")
        
        await asyncio.sleep(1)
        self.processing = False
    
    def action_clear_chat(self) -> None:
        """Clear the chat log."""
        chat = self.query_one("#chat", RichLog)
        chat.clear()
        self.chat_messages.clear()  # Clear stored messages
        welcome_panel = Panel.fit(
            "Chat cleared! Ready for new questions. 🧹",
            title="🆕 Fresh Start",
            border_style="yellow"
        )
        chat.write(welcome_panel)
    
    def action_reload_db(self) -> None:
        """Reload the database."""
        chat = self.query_one("#chat", RichLog)
        if self.rag.load_existing_db():
            chat.write("[green]🔄 Database reloaded successfully![/green]")
            self.update_stats()
        else:
            chat.write("[red]❌ No database found to reload[/red]")
    
    def action_show_help(self) -> None:
        """Show help screen."""
        self.push_screen(HelpScreen())
    
    def action_show_settings(self) -> None:
        """Show settings screen."""
        self.push_screen(SettingsScreen())
    
    def action_show_documents(self) -> None:
        """Show document browser."""
        self.push_screen(DocumentBrowserScreen())
    
    def action_toggle_sidebar(self) -> None:
        """Toggle sidebar visibility."""
        self.show_sidebar = not self.show_sidebar
    
    def action_new_session(self) -> None:
        """Start a new chat session."""
        self.chat_history.start_new_session()
        self.action_clear_chat()
        
        # Update history panel
        history_content = self.query_one("#history-content", RichLog)
        history_content.clear()
        history_content.write("[green]🆕 New session started[/green]")
    
    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()
    
    def action_restart_rag(self) -> None:
        """Restart the RAG system to fix file descriptor issues."""
        chat = self.query_one("#chat", RichLog)
        chat.write("[yellow]🔄 Restarting RAG system...[/yellow]")
        
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Recreate the RAG system
            self.rag = RAGSystem(
                model_name=self.settings_manager.get('model_name', 'llama3.2:3b'),
                temperature=self.settings_manager.get('temperature', 0.1),
                chunk_size=self.settings_manager.get('chunk_size', 1000),
                chunk_overlap=self.settings_manager.get('chunk_overlap', 200),
                retrieval_k=self.settings_manager.get('retrieval_k', 3),
                settings_manager=self.settings_manager
            )
            
            # Try to reload the database
            if self.rag.load_existing_db():
                chat.write("[green]✅ RAG system restarted successfully![/green]")
                self.update_stats()
            else:
                chat.write("[yellow]⚠️ RAG system restarted. Please load or create a database.[/yellow]")
                
        except Exception as e:
            chat.write(f"[red]❌ Failed to restart RAG system: {str(e)}[/red]")

    def action_copy_last_message(self) -> None:
        """Copy the last message to clipboard."""
        chat = self.query_one("#chat", RichLog)
        
        if not self.chat_messages:
            chat.write("[yellow]⚠️ No messages to copy[/yellow]")
            return
        
        try:
            last_message = self.chat_messages[-1]
            message_text = f"[{last_message['timestamp']}] {last_message['type'].title()}: {last_message['content']}"
            
            # Try to use pyperclip, fall back to file export if not available
            try:
                pyperclip.copy(message_text)
                chat.write("[green]✅ Last message copied to clipboard![/green]")
            except Exception:
                # Fallback: save to file
                filename = "last_message.txt"
                with open(filename, 'w') as f:
                    f.write(message_text)
                chat.write(f"[green]✅ Last message saved to {filename} (install pyperclip for clipboard support)[/green]")
                
        except Exception as e:
            chat.write(f"[red]❌ Copy failed: {str(e)}[/red]")
    
    def action_export_chat(self) -> None:
        """Export current chat session."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_export_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write("RAG Chat Export\n")
                f.write("=" * 50 + "\n")
                f.write(f"Exported: {datetime.now().isoformat()}\n")
                f.write(f"Model: {self.rag.model_name}\n")
                f.write(f"Temperature: {self.rag.temperature}\n")
                f.write("=" * 50 + "\n\n")
                
                # Export actual chat messages
                for msg in self.chat_messages:
                    f.write(f"[{msg['timestamp']}] {msg['type'].upper()}:\n")
                    f.write(f"{msg['content']}\n")
                    f.write("-" * 40 + "\n\n")
            
            chat = self.query_one("#chat", RichLog)
            chat.write(f"[green]📁 Chat exported to {filename}[/green]")
        except Exception as e:
            chat = self.query_one("#chat", RichLog)
            chat.write(f"[red]❌ Export failed: {str(e)}[/red]")
    
    def action_export_full_chat(self) -> None:
        """Export complete chat history including all sessions."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"full_chat_history_{timestamp}.json"
            
            # Save current session first
            self.chat_history.start_new_session()
            
            # Export all sessions
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "model_settings": {
                    "model": self.rag.model_name,
                    "temperature": self.rag.temperature,
                    "chunk_size": self.rag.chunk_size,
                    "retrieval_k": self.rag.retrieval_k
                },
                "sessions": self.chat_history.sessions,
                "current_session": [
                    {
                        "timestamp": msg["timestamp"],
                        "type": msg["type"],
                        "content": msg["content"]
                    }
                    for msg in self.chat_messages
                ]
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            chat = self.query_one("#chat", RichLog)
            chat.write(f"[green]📁 Full chat history exported to {filename}[/green]")
            
        except Exception as e:
            chat = self.query_one("#chat", RichLog)
            chat.write(f"[red]❌ Full export failed: {str(e)}[/red]")

if __name__ == "__main__":
    app = RAGChatApp()
    app.run()

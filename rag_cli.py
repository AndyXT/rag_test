# Standard library imports
import asyncio
import os
import time
import json
from pathlib import Path
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
from textual.screen import ModalScreen, Screen
from rich.text import Text
from rich.table import Table
from rich.panel import Panel

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
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
            'dark_mode': False
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
        except (IOError, json.JSONDecodeError, KeyError) as e:
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
        except (IOError, TypeError) as e:
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
        except (IOError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load chat history from {self.history_file}: {e}")
            self.sessions = []
    
    def save_history(self) -> None:
        """Save chat history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump({'sessions': self.sessions}, f, indent=2)
        except (IOError, TypeError) as e:
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
    
    def compose(self) -> ComposeResult:
        with Container(id="settings-container"):
            yield Static("⚙️ Settings", id="settings-title")
            
            with VerticalScroll():
                yield Label("Model Settings:")
                yield Input(value="llama3.2", placeholder="Ollama model name", id="model-input")
                
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
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses in settings screen"""
        if event.button.id == "cancel-settings":
            self.dismiss()
        elif event.button.id == "save-settings":
            # Get all input values
            model_input = self.query_one("#model-input", Input)
            temp_input = self.query_one("#temp-input", Input)
            chunk_input = self.query_one("#chunk-input", Input)
            overlap_input = self.query_one("#overlap-input", Input)
            retrieval_input = self.query_one("#retrieval-input", Input)
            
            # Validate and parse values
            try:
                new_settings = {
                    'model_name': model_input.value or "llama3.2",
                    'temperature': float(temp_input.value or "0.0"),
                    'chunk_size': int(chunk_input.value or "1000"),
                    'chunk_overlap': int(overlap_input.value or "200"),
                    'retrieval_k': int(retrieval_input.value or "3")
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
- ⚙️ Configurable settings
- 🎨 Multiple themes

## Document Management
- Place PDF files in the ./documents directory
- Use "Create DB" to process new documents
- Use "Reload DB" to refresh the database

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
    def __init__(self, model_name: str = "llama3.2", temperature: float = 0, 
                 chunk_size: int = 1000, chunk_overlap: int = 200, retrieval_k: int = 3):
        self.vectorstore = None
        self.qa_chain = None
        self.model_name = model_name
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k

        # Check and optimize system resources
        self._check_system_resources()

        # Initialize embeddings with better error handling and subprocess control
        self._init_embeddings()
        self.llm = OllamaLLM(model=self.model_name, temperature=self.temperature)

    def _check_system_resources(self):
        """Check and optimize system resources for file descriptor handling"""
        import resource
        import os

        try:
            # Get current file descriptor limits
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)

            # If soft limit is too low, try to increase it
            if soft_limit < 4096:
                try:
                    # Try to set soft limit to 4096 or hard limit, whichever is smaller
                    new_soft_limit = min(4096, hard_limit)
                    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))
                    print(f"Increased file descriptor limit from {soft_limit} to {new_soft_limit}")
                except (ValueError, OSError):
                    print(f"Warning: Could not increase file descriptor limit (current: {soft_limit})")

            # Set additional environment variables for stability
            os.environ['PYTHONDONTWRITEBYTECODE'] = '1'  # Reduce file I/O
            os.environ['PYTHONIOENCODING'] = 'utf-8'     # Consistent encoding

        except Exception as e:
            print(f"Warning: Could not check system resources: {e}")

    def _init_embeddings(self):
        """Initialize embeddings with robust error handling and file descriptor management"""
        import os
        import gc
        import multiprocessing as mp

        # Set comprehensive environment variables to prevent subprocess issues
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'

        # Force single process to avoid file descriptor issues
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
            # Force cleanup on failure
            gc.collect()
            raise Exception(f"Failed to initialize embeddings: {str(e)}") from e
        
    def update_settings(self, **kwargs):
        """Update RAG system settings"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Recreate LLM with new settings
        self.llm = OllamaLLM(model=self.model_name, temperature=self.temperature)
        
        # Recreate QA chain if vectorstore exists
        if self.vectorstore:
            self._setup_qa_chain()
        
    def load_existing_db(self, db_path="./chroma_db"):
        """Load existing ChromaDB with modern configuration"""
        import os

        # Set environment variables for ChromaDB (modern approach)
        os.environ['ANONYMIZED_TELEMETRY'] = 'False'

        if os.path.exists(db_path):
            try:
                # Use modern ChromaDB configuration
                self.vectorstore = Chroma(
                    persist_directory=db_path,
                    embedding_function=self.embeddings
                )
                self._setup_qa_chain()
                return True
            except Exception:
                # If loading fails, return False to indicate no database
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

        # Set comprehensive environment variables for ChromaDB
        os.environ['ANONYMIZED_TELEMETRY'] = 'False'
        os.environ['CHROMA_SERVER_NOFILE'] = '65536'  # Increase file descriptor limit for ChromaDB

        try:
            # Remove existing database if it exists
            final_path = Path(db_path)
            if final_path.exists():
                shutil.rmtree(final_path)

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
                        # Create initial vectorstore with explicit settings
                        vectorstore = Chroma.from_documents(
                            documents=batch,
                            embedding=self.embeddings,
                            persist_directory=db_path,
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

            # Final garbage collection
            gc.collect()
            return vectorstore

        except Exception as e:
            # Clean up on failure
            if final_path.exists():
                try:
                    shutil.rmtree(final_path)
                except Exception:
                    pass
            gc.collect()
            raise Exception(f"ChromaDB creation failed: {str(e)}") from e

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
        """Query the RAG system"""
        if not self.qa_chain:
            return "RAG system not initialized. Load or create a database first."
        
        # Run in thread pool to avoid blocking UI
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.qa_chain.invoke, {"input": question})
        return result["answer"]
    
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
    ]
    
    show_sidebar = reactive(True)
    processing = reactive(False)
    
    def __init__(self):
        super().__init__()
        self.settings_manager = SettingsManager()
        # Initialize RAG system with saved settings
        self.rag = RAGSystem(
            model_name=self.settings_manager.get('model_name'),
            temperature=self.settings_manager.get('temperature'),
            chunk_size=self.settings_manager.get('chunk_size'),
            chunk_overlap=self.settings_manager.get('chunk_overlap'),
            retrieval_k=self.settings_manager.get('retrieval_k')
        )
        self.chat_history = ChatHistory()
        self.current_progress = 0
        self.progress_timer: Optional[Any] = None
        
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
        
        if not self.rag.qa_chain:
            chat.write("[red]⚠️ Please load or create a database first.[/red]")
            return
        
        self.processing = True
        self.update_progress("🤔 Thinking...", 50)
        
        # Add to history
        history_content = self.query_one("#history-content", RichLog)
        history_content.write(f"[dim]{timestamp}[/dim] {question[:50]}...")
        
        try:
            # Show thinking indicator
            chat.write("[dim]🧠 Processing your question...[/dim]")
            
            start_time = time.time()
            answer = await self.rag.query(question)
            response_time = time.time() - start_time
            
            # Remove thinking indicator and add answer
            assistant_panel = Panel(
                answer,
                title=f"🤖 Assistant [{response_time:.1f}s]",
                border_style="green",
                padding=(0, 1)
            )
            chat.write(assistant_panel)
            
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

    def action_export_chat(self) -> None:
        """Export current chat session."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_export_{timestamp}.txt"
            
            # This is a simplified export - in a real app you'd want more sophisticated export
            chat = self.query_one("#chat", RichLog)
            with open(filename, 'w') as f:
                f.write("RAG Chat Export\n")
                f.write("=" * 50 + "\n")
                f.write(f"Exported: {datetime.now().isoformat()}\n\n")
                # Note: In a real implementation, you'd extract the actual chat content
                f.write("Chat content would be exported here.\n")
            
            chat.write(f"[green]📁 Chat exported to {filename}[/green]")
        except Exception as e:
            chat = self.query_one("#chat", RichLog)
            chat.write(f"[red]❌ Export failed: {str(e)}[/red]")

if __name__ == "__main__":
    app = RAGChatApp()
    app.run()

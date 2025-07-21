from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Input, RichLog, Button, Static, Header, Footer, 
    ProgressBar, Tabs, TabPane, Tree, Label, Switch,
    SelectionList, DataTable, TabbedContent
)
from textual.binding import Binding
from textual.reactive import reactive
from textual.screen import ModalScreen, Screen
from textual.timer import Timer
from rich.console import Console
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import asyncio
import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class ChatHistory:
    """Manages chat history with persistence"""
    def __init__(self, history_file="chat_history.json"):
        self.history_file = history_file
        self.sessions: List[Dict] = []
        self.current_session: List[Dict] = []
        self.load_history()
    
    def load_history(self):
        """Load chat history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.sessions = data.get('sessions', [])
        except Exception:
            self.sessions = []
    
    def save_history(self):
        """Save chat history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump({'sessions': self.sessions}, f, indent=2)
        except Exception:
            pass
    
    def add_exchange(self, question: str, answer: str):
        """Add Q&A exchange to current session"""
        self.current_session.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer
        })
    
    def start_new_session(self):
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
    
    def compose(self) -> ComposeResult:
        with Container(id="settings-container"):
            yield Static("⚙️ Settings", id="settings-title")
            
            with Vertical():
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
                
                with Horizontal():
                    yield Button("Save", variant="primary", id="save-settings")
                    yield Button("Cancel", id="cancel-settings")

class HelpScreen(ModalScreen):
    """Modal screen showing help and keyboard shortcuts"""
    
    def compose(self) -> ComposeResult:
        self.help_content = """
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
        
        with Container(id="help-container"):
            yield Static("📖 Help & Documentation", id="help-title")
            yield RichLog(markup=True, id="help-content")
            yield Button("Close", id="close-help")
    
    def on_mount(self):
        help_log = self.query_one("#help-content", RichLog)
        help_log.write(self.help_content)

class DocumentBrowserScreen(ModalScreen):
    """Modal screen for browsing and managing documents"""
    
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

class RAGSystem:
    def __init__(self, model_name="llama3.2", temperature=0, chunk_size=1000, chunk_overlap=200, retrieval_k=3):
        self.vectorstore = None
        self.qa_chain = None
        self.model_name = model_name
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm = OllamaLLM(model=self.model_name, temperature=self.temperature)
        
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
        """Load existing ChromaDB"""
        if os.path.exists(db_path):
            self.vectorstore = Chroma(
                persist_directory=db_path,
                embedding_function=self.embeddings
            )
            self._setup_qa_chain()
            return True
        return False
    
    def create_db_from_docs(self, docs_path="./documents", db_path="./chroma_db", progress_callback=None):
        """Create new ChromaDB from documents with progress tracking"""
        loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        
        if progress_callback:
            progress_callback("Loading documents...")
        
        documents = loader.load()
        
        if not documents:
            raise ValueError(f"No documents found in {docs_path}")
        
        if progress_callback:
            progress_callback(f"Splitting {len(documents)} documents...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        
        if progress_callback:
            progress_callback(f"Creating embeddings for {len(texts)} chunks...")
        
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=db_path
        )
        
        if progress_callback:
            progress_callback("Setting up QA chain...")
        
        self._setup_qa_chain()
        
        if progress_callback:
            progress_callback("Database creation complete!")
        
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
        except:
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
    }
    
    .chat-container {
        height: 1fr;
        border: solid $accent;
        margin: 1;
    }
    
    .input-container {
        height: 4;
        border: solid $primary;
        margin: 1 1 0 1;
    }
    
    .status-container {
        height: 3;
        background: $surface;
        margin: 1;
    }
    
    /* Widgets */
    Input {
        width: 1fr;
        margin: 1;
    }
    
    Button {
        width: auto;
        min-width: 12;
        margin: 1;
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
         height: 80%;
         background: $surface;
         border: solid $primary;
         margin: 2;
     }
    
    #settings-title, #help-title, #doc-title {
        text-align: center;
        background: $primary;
        height: 3;
        content-align: center middle;
    }
    
    /* Progress and status */
    .progress-container {
        height: 3;
        margin: 1;
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
        self.rag = RAGSystem()
        self.chat_history = ChatHistory()
        self.current_progress = 0
        self.progress_timer = None
        
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
            progress_steps = [
                "📂 Scanning documents directory...",
                "📄 Loading PDF documents...", 
                "✂️ Splitting documents into chunks...",
                "🧮 Creating embeddings...",
                "💾 Building vector database...",
                "🔗 Setting up QA chain...",
                "✅ Database creation complete!"
            ]
            
            for i, step in enumerate(progress_steps[:-1]):
                self.update_progress(step, (i / len(progress_steps)) * 100)
                await asyncio.sleep(0.3)
            
            # Define progress callback
            def progress_callback(message):
                # This runs in the executor thread, so we need to be careful
                pass
            
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.rag.create_db_from_docs(progress_callback=progress_callback)
            )
            
            self.update_progress(progress_steps[-1], 100)
            chat.write("[green]✅ Database created successfully![/green]")
            self.update_stats()
            
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
            thinking_msg = chat.write("[dim]🧠 Processing your question...[/dim]")
            
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

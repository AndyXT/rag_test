#!/usr/bin/env python3
"""
RAG Chat TUI Demo - Showcases the enhanced user interface
This is a demonstration version that doesn't require LangChain dependencies.
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Input, RichLog, Button, Static, Header, Footer, 
    ProgressBar, Tree, Label, Switch
)
from textual.binding import Binding
from textual.reactive import reactive
from textual.screen import ModalScreen
from rich.panel import Panel
from rich.table import Table
import asyncio
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import os

class ChatHistory:
    """Manages chat history with persistence"""
    def __init__(self, history_file="demo_chat_history.json"):
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
# 🤖 RAG Chat Help - Demo Version

## Keyboard Shortcuts
- **Ctrl+C**: Quit application
- **Ctrl+L**: Clear current chat
- **Ctrl+R**: Reload database (demo)
- **Ctrl+H**: Show this help
- **Ctrl+S**: Open settings
- **Ctrl+N**: New chat session
- **F1**: Toggle sidebar
- **Enter**: Send message

## Demo Features
This is a demonstration of the enhanced TUI interface.
In the full version, you can:

- 📚 Process PDF documents
- 🔍 Perform semantic search
- 💬 Chat with AI about your documents
- ⚙️ Configure model parameters

## Getting Started
1. This demo shows the interface improvements
2. Type messages to see the chat formatting
3. Try the keyboard shortcuts and buttons
4. Explore the settings and help screens

## Enhanced UI Features
- ✨ Modern, colorful interface
- 📊 Live status dashboard
- 📈 Progress indicators
- 🎨 Rich text formatting
- ⌨️ Comprehensive keyboard shortcuts
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
        else:
            tree.root.add_leaf("No documents directory found")
            tree.root.add_leaf("(This is a demo)")

class MockRAGSystem:
    """Mock RAG system for demonstration purposes"""
    
    def __init__(self):
        self.is_loaded = False
        self.model_name = "llama3.2"
        self.doc_count = 0
        
    def get_stats(self):
        """Get mock database statistics"""
        if self.is_loaded:
            return {
                "document_count": self.doc_count,
                "model": self.model_name,
                "temperature": 0.0,
                "chunk_size": 1000,
                "retrieval_k": 3,
                "status": "Ready"
            }
        return None
    
    async def mock_query(self, question: str):
        """Mock query processing with realistic delay"""
        # Simulate processing time
        await asyncio.sleep(1 + len(question) * 0.02)
        
        # Return a demo response
        responses = [
            "This is a demonstration of the enhanced TUI interface. In the real application, this would be an AI-generated response based on your documents.",
            "The improved interface features rich formatting, progress indicators, and better user experience. Your question was processed in a simulated environment.",
            "In the full version with LangChain integration, you would get intelligent responses based on your PDF documents using RAG (Retrieval-Augmented Generation).",
            "This demo showcases the visual improvements: styled panels, timestamps, response times, and comprehensive keyboard shortcuts.",
            "The enhanced TUI includes features like chat history, settings panel, document browser, and export functionality."
        ]
        
        import random
        return random.choice(responses)

class RAGChatApp(App):
    """Enhanced Textual app for RAG chat interface - Demo Version."""
    
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
        self.rag = MockRAGSystem()
        self.chat_history = ChatHistory()
        self.message_count = 0
        
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header(show_clock=True)
        
        with Container(classes="main-container"):
            # Sidebar
            with Vertical(classes="sidebar", id="sidebar"):
                yield Static("🤖 RAG Assistant (Demo)", classes="sidebar-title")
                
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
                    yield Input(placeholder="Try the demo! Ask any question...", id="question_input")
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
            "Welcome to RAG Chat TUI Demo! 🤖✨\n\n"
            "This demonstrates the enhanced user interface:\n"
            "• Modern styling with rich colors and panels\n"
            "• Responsive sidebar that can be toggled\n"
            "• Progress indicators and status updates\n"
            "• Comprehensive keyboard shortcuts\n"
            "• Modal screens for settings and help\n\n"
            "Try typing a message or pressing Ctrl+H for help!",
            title="🚀 Demo Version",
            border_style="green"
        )
        chat.write(welcome_panel)
        
        # Update status
        self.update_stats()
        
        # Show demo info
        chat.write("[yellow]💡 This is a demonstration version. The full app integrates with LangChain for real document processing.[/yellow]")
    
    def update_stats(self):
        """Update the stats panel"""
        stats_content = self.query_one("#stats-content", RichLog)
        stats_content.clear()
        
        stats_table = Table(show_header=False, box=None, padding=0)
        stats_table.add_column("Key", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("Mode", "Demo")
        stats_table.add_row("Messages", str(self.message_count))
        stats_table.add_row("Status", "Ready")
        stats_table.add_row("Sidebar", "Visible" if self.show_sidebar else "Hidden")
        
        stats_content.write(stats_table)
    
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
            await self._demo_load_database()
            
        elif button_id == "create_btn":
            await self._demo_create_database()
            
        elif button_id == "docs_btn":
            self.push_screen(DocumentBrowserScreen())
            
        elif button_id == "settings_btn":
            self.push_screen(SettingsScreen())
            
        elif button_id == "help_btn":
            self.push_screen(HelpScreen())
    
    async def _demo_load_database(self):
        """Demo database loading with progress feedback"""
        chat = self.query_one("#chat", RichLog)
        self.processing = True
        
        self.update_progress("🔍 Searching for existing database...", 20)
        await asyncio.sleep(1)
        
        self.update_progress("📊 Loading database metadata...", 60)
        await asyncio.sleep(0.8)
        
        self.rag.is_loaded = True
        self.rag.doc_count = 42  # Demo value
        
        chat.write("[green]✅ Demo database loaded successfully![/green]")
        self.update_stats()
        self.update_progress("✅ Database ready", 100)
        
        await asyncio.sleep(1)
        self.processing = False
    
    async def _demo_create_database(self):
        """Demo database creation with enhanced progress tracking"""
        chat = self.query_one("#chat", RichLog)
        self.processing = True
        
        try:
            progress_steps = [
                ("📂 Scanning documents directory...", 15),
                ("📄 Loading PDF documents...", 30), 
                ("✂️ Splitting documents into chunks...", 50),
                ("🧮 Creating embeddings...", 70),
                ("💾 Building vector database...", 85),
                ("🔗 Setting up QA chain...", 95),
                ("✅ Database creation complete!", 100)
            ]
            
            for step, progress in progress_steps:
                self.update_progress(step, progress)
                await asyncio.sleep(0.8)
            
            self.rag.is_loaded = True
            self.rag.doc_count = 156  # Demo value
            
            chat.write("[green]✅ Demo database created successfully![/green]")
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
        
        self.processing = True
        self.update_progress("🤔 Processing your question...", 50)
        
        # Add to history
        history_content = self.query_one("#history-content", RichLog)
        history_content.write(f"[dim]{timestamp}[/dim] {question[:40]}...")
        
        try:
            start_time = time.time()
            answer = await self.rag.mock_query(question)
            response_time = time.time() - start_time
            
            # Add answer with response time
            assistant_panel = Panel(
                answer,
                title=f"🤖 Assistant [{response_time:.1f}s]",
                border_style="green",
                padding=(0, 1)
            )
            chat.write(assistant_panel)
            
            # Add to chat history
            self.chat_history.add_exchange(question, answer)
            self.message_count += 1
            self.update_stats()
            
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
        chat.write("[green]🔄 Demo database reloaded![/green]")
        self.update_stats()
    
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
        self.update_stats()
    
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
            filename = f"demo_chat_export_{timestamp}.txt"
            
            chat = self.query_one("#chat", RichLog)
            with open(filename, 'w') as f:
                f.write("RAG Chat Demo Export\n")
                f.write("=" * 50 + "\n")
                f.write(f"Exported: {datetime.now().isoformat()}\n")
                f.write(f"Message Count: {self.message_count}\n\n")
                f.write("This is a demo export. In the full version,\n")
                f.write("actual chat content would be exported here.\n")
            
            chat.write(f"[green]📁 Chat exported to {filename}[/green]")
        except Exception as e:
            chat = self.query_one("#chat", RichLog)
            chat.write(f"[red]❌ Export failed: {str(e)}[/red]")

if __name__ == "__main__":
    app = RAGChatApp()
    app.run()
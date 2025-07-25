# RAG Chat Application UI
import asyncio
import json
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

# Rich formatting imports
from rich.table import Table
from rich.panel import Panel

# Textual framework imports
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Static, Input, RichLog, Header, Footer, ProgressBar
from textual.binding import Binding
from textual.reactive import reactive

# Internal imports
from rag_cli.core.settings_manager import SettingsManager
from rag_cli.core.chat_history import ChatHistory
from rag_cli.services.rag_service import RAGService
from rag_cli.ui.screens.settings_screen import SettingsScreen
from rag_cli.ui.screens.help_screen import HelpScreen
from rag_cli.ui.screens.document_browser import DocumentBrowserScreen
from rag_cli.utils.constants import PYPERCLIP_AVAILABLE, pyperclip


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
    
    .ollama-field {
        display: none;
    }
    
    .api-field.visible,
    .openai-field.visible,
    .anthropic-field.visible,
    .ollama-field.visible {
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
        Binding("ctrl+t", "toggle_context", "Toggle Context"),
        Binding("ctrl+shift+c", "clean_cache", "Clean Cache"),
    ]

    show_sidebar = reactive(True)
    processing = reactive(False)

    def __init__(self) -> None:
        super().__init__()
        self.settings_manager = SettingsManager()
        # Initialize RAG service
        self.rag_service = RAGService(settings_file="settings.json")
        self.chat_history = ChatHistory()
        self.current_progress = 0
        self.progress_timer: Optional[Any] = None
        self.chat_messages: List[Dict[str, Any]] = []  # Store messages for copying  # Store messages for copying

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
                    yield Input(
                        placeholder="Ask a question about your documents...",
                        id="question_input",
                    )
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
            border_style="green",
        )
        chat.write(welcome_panel)

        # Update status
        self.update_stats()

        # Try to load existing DB automatically
        if self.rag_service.load_database():
            chat.write("[green]✓ Automatically loaded existing ChromaDB[/green]")
            self.update_stats()
        else:
            chat.write(
                "[yellow]💡 No existing database found. Create one from documents or load an existing one.[/yellow]"
            )

    def update_stats(self):
        """Update the stats panel"""
        stats_content = self.query_one("#stats-content", RichLog)
        stats_content.clear()

        info = self.rag_service.get_system_info()
        if info:
            stats_table = Table(show_header=False, box=None, padding=0)
            stats_table.add_column("Key", style="cyan")
            stats_table.add_column("Value", style="white")

            # Extract key information from nested structure
            db_info = info.get("database", {})
            settings = info.get("settings", {})
            system = info.get("system", {})
            
            # Display most important stats
            if db_info.get("loaded") and db_info.get("document_count", 0) > 0:
                stats_table.add_row("Documents", str(db_info.get("document_count", "Unknown")))
            stats_table.add_row("Model", settings.get("model", "Unknown"))
            stats_table.add_row("Temperature", f"{settings.get('temperature', 0.1):.1f}")
            stats_table.add_row("Chunk Size", str(settings.get("chunk_size", 1000)))
            stats_table.add_row("DB Loaded", "✅" if system.get("vectorstore_loaded") else "❌")

            stats_content.write(stats_table)
        else:
            stats_content.write("[dim]No database loaded[/dim]")

    def update_progress(self, message: str, progress: float | None = None):
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

        if self.rag_service.load_database():
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
        from rag_cli.services.document_service import DocumentService
        
        chat = self.query_one("#chat", RichLog)
        self.processing = True

        try:
            # Validate documents directory
            pdf_files = await self._validate_documents_directory()
            
            # Show estimated processing time
            _, time_str = DocumentService.estimate_processing_time(pdf_files)
            self.add_message("info", f"Estimated processing time: {time_str}")
            
            # Check disk space
            has_space, space_msg = DocumentService.check_disk_space("./chroma_db")
            if not has_space:
                raise ValueError(space_msg)
                
            await self._create_database_with_progress(pdf_files, chat)
        except Exception as e:
            await self._handle_database_creation_error(e, chat)
        finally:
            await asyncio.sleep(1.5)
            self.processing = False

    async def _validate_documents_directory(self):
        """Validate documents directory and return PDF files"""
        from rag_cli.services.document_service import DocumentService
        
        is_valid, message, pdf_files = DocumentService.validate_documents_directory("./documents")
        
        if not is_valid:
            raise ValueError(message)
        
        return pdf_files

    async def _create_database_with_progress(self, pdf_files, chat):
        """Create database with progress tracking"""
        self.update_progress(f"📂 Found {len(pdf_files)} PDF files...", 10)
        await asyncio.sleep(0.5)

        # Define a thread-safe progress callback that updates the UI
        progress_messages = []

        def progress_callback(message):
            progress_messages.append(message)

        # Run database creation in thread executor to avoid async context issues
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.rag_service.create_database(
                docs_path="./documents",
                db_path="./chroma_db",
                progress_callback=progress_callback,
            ),
        )

        self.update_progress("✅ Database creation complete!", 100)
        chat.write("[green]✅ Database created successfully![/green]")

        # Show progress messages that were collected
        if progress_messages:
            chat.write(
                f"[blue]📊 Processing summary: {progress_messages[-1]}[/blue]"
            )

        # Show summary of what was processed
        stats = self.rag_service.get_system_info()
        if stats and stats.get("document_count", 0) > 0:
            chat.write(
                f"[blue]📊 Processed {stats['document_count']} document chunks[/blue]"
            )

        self.update_stats()

    async def _handle_database_creation_error(self, error, chat):
        """Handle database creation errors with specific messages"""
        error_msg = str(error)
        
        error_handlers = {
            "all batches failed": self._handle_batch_failure_error,
            "fds_to_keep": self._handle_file_descriptor_error,
            "file descriptor": self._handle_file_descriptor_error,
            "No PDF files found": self._handle_no_pdf_error,
            "No documents could be processed": self._handle_processing_error
        }
        
        for key, handler in error_handlers.items():
            if key in error_msg or key.lower() in error_msg.lower():
                handler(chat)
                self.update_progress(f"❌ Error: {error_msg[:50]}...")
                return
        
        # Default error handling
        chat.write(f"[red]❌ Database creation error: {error_msg}[/red]")
        chat.write(
            "[yellow]💡 Check the error message above for specific details.[/yellow]"
        )
        self.update_progress(f"❌ Error: {error_msg[:50]}...")

    def _handle_batch_failure_error(self, chat):
        """Handle batch failure errors"""
        chat.write(
            "[red]❌ Database creation failed: All document batches failed to process[/red]"
        )
        chat.write(
            "[yellow]💡 This may be due to embedding model issues or document format problems.[/yellow]"
        )
        chat.write(
            "[yellow]💡 Try restarting the application or check the terminal for detailed errors.[/yellow]"
        )

    def _handle_file_descriptor_error(self, chat):
        """Handle file descriptor errors"""
        chat.write(
            "[red]❌ PDF processing error: File descriptor issue[/red]"
        )
        chat.write(
            "[yellow]💡 This is often caused by corrupted PDFs or system limitations.[/yellow]"
        )
        chat.write(
            "[yellow]💡 Try removing problematic PDF files or restarting the application.[/yellow]"
        )

    def _handle_no_pdf_error(self, chat):
        """Handle no PDF files error"""
        chat.write(
            "[red]❌ No PDF files found in ./documents directory[/red]"
        )
        chat.write(
            "[yellow]💡 Please add some PDF files to the ./documents directory.[/yellow]"
        )

    def _handle_processing_error(self, chat):
        """Handle processing errors"""
        chat.write("[red]❌ All PDF files failed to process[/red]")
        chat.write(
            "[yellow]💡 Check if your PDF files are corrupted or password-protected.[/yellow]"
        )

    async def _validate_question(self, question: str, chat) -> bool:
        """Validate question and system state."""
        if not question.strip():
            return False
            
        if not self.rag_service.rag_system.vectorstore:
            chat.write("[red]⚠️ Please load or create a database first.[/red]")
            return False
            
        return True
    
    def _display_user_question(self, question: str, chat) -> str:
        """Display user question with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        user_panel = Panel(
            question, title=f"👤 You [{timestamp}]", border_style="blue", padding=(0, 1)
        )
        chat.write(user_panel)
        
        # Store the question for copying
        self.chat_messages.append(
            {"type": "user", "content": question, "timestamp": timestamp}
        )
        
        return timestamp
    
    async def _execute_query(self, question: str) -> dict:
        """Execute the query with timeout."""
        try:
            result = await asyncio.wait_for(
                self.rag_service.query_service.process_query(question), 
                timeout=60.0
            )
            return result
        except asyncio.TimeoutError:
            raise Exception("Query timed out after 60 seconds. Try disabling query expansion or reranking in settings.")
    
    def _display_context(self, context_docs: list, chat) -> None:
        """Display context documents if enabled."""
        show_context = self.settings_manager.get("show_context", False)
        
        if not show_context or not context_docs:
            return
            
        try:
            context_content = []
            for i, doc in enumerate(context_docs):
                doc_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                # Truncate long content for display
                if len(doc_content) > 500:
                    doc_content = doc_content[:500] + "..."
                context_content.append(f"[bold]Document {i+1}:[/bold]\n{doc_content}")
            
            if context_content:
                context_text = "\n\n".join(context_content)
                context_panel = Panel(
                    context_text,
                    title=f"📚 Retrieved Context ({len(context_docs)} documents)",
                    border_style="cyan",
                    padding=(1, 1),
                )
                chat.write(context_panel)
        except Exception as e:
            chat.write(f"[red]Error displaying context: {str(e)}[/red]")
    
    def _display_answer(self, answer: str, response_time: float, chat) -> None:
        """Display the assistant's answer."""
        assistant_panel = Panel(
            answer,
            title=f"🤖 Assistant [{response_time:.1f}s]",
            border_style="green",
            padding=(0, 1),
        )
        chat.write(assistant_panel)
        
        # Store the answer for copying
        self.chat_messages.append(
            {
                "type": "assistant",
                "content": answer,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        )
    
    async def _handle_query_error(self, error: Exception, chat) -> None:
        """Handle and display query errors."""
        error_panel = Panel(
            f"Sorry, I encountered an error: {str(error)}",
            title="❌ Error",
            border_style="red",
            padding=(0, 1),
        )
        chat.write(error_panel)
        self.update_progress(f"❌ Error: {str(error)}")

    async def _process_question(self, question: str) -> None:
        """Process user question with enhanced UI feedback."""
        chat = self.query_one("#chat", RichLog)
        
        # Validate question and system state
        if not await self._validate_question(question, chat):
            return
        
        # Display user question
        timestamp = self._display_user_question(question, chat)
        
        # Update UI state
        self.processing = True
        self.update_progress("🤔 Thinking...", 50)
        
        # Add to history
        history_content = self.query_one("#history-content", RichLog)
        history_content.write(f"[dim]{timestamp}[/dim] {question[:50]}...")
        
        try:
            # Show processing indicator
            chat.write("[dim]🧠 Processing your question...[/dim]")
            
            # Execute query
            start_time = time.time()
            result = await self._execute_query(question)
            response_time = time.time() - start_time
            
            # Extract response and context
            answer = result.get("response", "No response") if isinstance(result, dict) else str(result)
            context_docs = result.get("context", []) if isinstance(result, dict) else []
            
            # Display context if enabled
            self._display_context(context_docs, chat)
            
            # Display answer
            self._display_answer(answer, response_time, chat)
            
            # Update chat history
            self.chat_history.add_exchange(question, answer)
            self.update_progress("✅ Response generated", 100)
            
        except Exception as e:
            await self._handle_query_error(e, chat)
        finally:
            # Reset UI state
            await asyncio.sleep(1)
            self.processing = False
            self.update_progress("", 0)

    def action_clear_chat(self) -> None:
        """Clear the chat log."""
        chat = self.query_one("#chat", RichLog)
        chat.clear()
        self.chat_messages.clear()  # Clear stored messages
        welcome_panel = Panel.fit(
            "Chat cleared! Ready for new questions. 🧹",
            title="🆕 Fresh Start",
            border_style="yellow",
        )
        chat.write(welcome_panel)

    def action_reload_db(self) -> None:
        """Reload the database."""
        chat = self.query_one("#chat", RichLog)
        if self.rag_service.load_database():
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

    def action_toggle_context(self) -> None:
        """Toggle context display setting."""
        current_value = self.settings_manager.get("show_context", False)
        new_value = not current_value
        self.settings_manager.save_settings({"show_context": new_value})
        
        chat = self.query_one("#chat", RichLog)
        status = "enabled" if new_value else "disabled"
        chat.write(f"[blue]ℹ Context display {status}[/blue]")

    def action_clean_cache(self) -> None:
        """Clean HuggingFace cache to fix embedding issues."""
        chat = self.query_one("#chat", RichLog)
        chat.write("[yellow]🧹 Cleaning HuggingFace cache...[/yellow]")
        
        try:
            # Clean the cache
            self.rag_service.database_service.cache_manager.clean_hf_cache_locks()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            chat.write("[green]✅ Cache cleaned successfully! You may want to reload the database (Ctrl+R).[/green]")
        except Exception as e:
            chat.write(f"[red]❌ Failed to clean cache: {str(e)}[/red]")

    def action_new_session(self) -> None:
        """Start a new chat session."""
        self.chat_history.start_new_session()
        self.action_clear_chat()

        # Update history panel
        history_content = self.query_one("#history-content", RichLog)
        history_content.clear()
        history_content.write("[green]🆕 New session started[/green]")

    async def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_restart_rag(self) -> None:
        """Restart the RAG system completely."""
        chat = self.query_one("#chat", RichLog)
        chat.write("[yellow]🔄 Restarting RAG system...[/yellow]")

        try:
            # Force garbage collection
            import gc
            gc.collect()

            # Reset the service
            self.rag_service.reset_system()
            
            # Reinitialize the service with current settings
            self.rag_service = RAGService(settings_file="settings.json")

            # Try to reload the database
            if self.rag_service.load_database():
                chat.write("[green]✅ RAG system restarted successfully![/green]")
                self.update_stats()
            else:
                chat.write(
                    "[yellow]⚠️ RAG system restarted. Please load or create a database.[/yellow]"
                )

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
            if PYPERCLIP_AVAILABLE and pyperclip is not None:
                try:
                    pyperclip.copy(message_text)
                    chat.write("[green]✅ Last message copied to clipboard![/green]")
                except Exception:
                    # Fallback: save to file
                    filename = "last_message.txt"
                    with open(filename, "w") as f:
                        f.write(message_text)
                    chat.write(
                        f"[green]✅ Last message saved to {filename} (clipboard failed)[/green]"
                    )
            else:
                # pyperclip not available, save to file
                filename = "last_message.txt"
                with open(filename, "w") as f:
                    f.write(message_text)
                chat.write(
                    f"[green]✅ Last message saved to {filename} (install pyperclip for clipboard support)[/green]"
                )

        except Exception as e:
            chat.write(f"[red]❌ Copy failed: {str(e)}[/red]")

    def action_export_chat(self) -> None:
        """Export current chat session."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_export_{timestamp}.txt"

            with open(filename, "w") as f:
                f.write("RAG Chat Export\n")
                f.write("=" * 50 + "\n")
                f.write(f"Exported: {datetime.now().isoformat()}\n")
                f.write(f"Model: {self.rag_service.settings_manager.get("model_name", "unknown")}\n")
                f.write(f"Temperature: {self.rag_service.settings_manager.get("temperature", 0.1)}\n")
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
                    "model": self.rag_service.settings_manager.get("model_name", "unknown"),
                    "temperature": self.rag_service.settings_manager.get("temperature", 0.1),
                    "chunk_size": self.rag_service.settings_manager.get("chunk_size", 1000),
                    "retrieval_k": self.rag_service.settings_manager.get("retrieval_k", 3),
                },
                "sessions": self.chat_history.sessions,
                "current_session": [
                    {
                        "timestamp": msg["timestamp"],
                        "type": msg["type"],
                        "content": msg["content"],
                    }
                    for msg in self.chat_messages
                ],
            }

            with open(filename, "w") as f:
                json.dump(export_data, f, indent=2)

            chat = self.query_one("#chat", RichLog)
            chat.write(f"[green]📁 Full chat history exported to {filename}[/green]")

        except Exception as e:
            chat = self.query_one("#chat", RichLog)
            chat.write(f"[red]❌ Full export failed: {str(e)}[/red]")


if __name__ == "__main__":
    from rag_cli.utils.logger import RichLogger
    RichLogger.set_tui_mode(True)
    app = RAGChatApp()
    app.run()
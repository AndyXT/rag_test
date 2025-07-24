"""Help screen for RAG CLI application"""
from typing import Any
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Button, Static, RichLog


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
- **Ctrl+Shift+C**: Clean cache (fixes retrieval issues)
- **Ctrl+T**: Toggle context display
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

    async def action_dismiss(self, result: Any | None = None) -> None:
        """Handle escape key to close modal"""
        self.dismiss()

    def action_quit(self) -> None:
        """Quit the application from help screen."""
        app = self.app  # type: ignore
        app.exit()
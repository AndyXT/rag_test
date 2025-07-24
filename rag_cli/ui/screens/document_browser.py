"""Document browser screen for RAG CLI application"""
from pathlib import Path
from typing import Any
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, Tree, Label, RichLog


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

    async def action_dismiss(self, result: Any | None = None) -> None:
        """Handle escape key to close modal"""
        self.dismiss()

    def action_quit(self) -> None:
        """Quit the application from document browser."""
        app = self.app  # type: ignore
        app.exit()
"""RAG CLI - A Python RAG (Retrieval-Augmented Generation) chat application with TUI"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Core components
from rag_cli.core.settings_manager import SettingsManager
from rag_cli.core.chat_history import ChatHistory
from rag_cli.core.rag_system import RAGSystem

# UI components
from rag_cli.ui.app import RAGChatApp
from rag_cli.ui.screens.settings_screen import SettingsScreen
from rag_cli.ui.screens.help_screen import HelpScreen
from rag_cli.ui.screens.document_browser import DocumentBrowserScreen

# Main entry point
from rag_cli.main import main

__all__ = [
    "SettingsManager",
    "ChatHistory",
    "RAGSystem",
    "RAGChatApp",
    "SettingsScreen",
    "HelpScreen",
    "DocumentBrowserScreen",
    "main",
]
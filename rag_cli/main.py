#!/usr/bin/env python3
"""Main entry point for RAG CLI application"""

# Import constants first to set up environment variables
import rag_cli.utils.constants  # This ensures environment setup runs
from rag_cli.ui.app import RAGChatApp
from rag_cli.utils.logger import RichLogger


def main():
    """Run the RAG Chat application"""
    # Enable TUI mode to prevent duplicate console output
    RichLogger.set_tui_mode(True)
    
    app = RAGChatApp()
    app.run()


if __name__ == "__main__":
    main()
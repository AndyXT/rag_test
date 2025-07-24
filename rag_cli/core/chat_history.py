"""Chat history management for RAG CLI application"""
import os
import json
from datetime import datetime
from typing import List, Dict, Any


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
                with open(self.history_file, "r") as f:
                    data = json.load(f)
                    self.sessions = data.get("sessions", [])
        except (
            IOError,
            OSError,
            PermissionError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            AttributeError,
        ) as e:
            print(f"Warning: Could not load chat history from {self.history_file}: {e}")
            self.sessions = []

    def save_history(self) -> None:
        """Save chat history to file"""
        try:
            with open(self.history_file, "w") as f:
                json.dump({"sessions": self.sessions}, f, indent=2)
        except (IOError, OSError, PermissionError, TypeError) as e:
            print(f"Warning: Could not save chat history to {self.history_file}: {e}")

    def add_exchange(self, question: str, answer: str) -> None:
        """Add Q&A exchange to current session"""
        self.current_session.append(
            {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": answer,
            }
        )

    def start_new_session(self) -> None:
        """Start a new chat session"""
        if self.current_session:
            session_data = {
                "start_time": datetime.now().isoformat(),
                "exchanges": self.current_session.copy(),
            }
            self.sessions.append(session_data)
            self.save_history()
        self.current_session = []
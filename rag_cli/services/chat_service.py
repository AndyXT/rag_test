"""Service for handling chat operations and history"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from rag_cli.utils.logger import RichLogger


class ChatService:
    """Handles chat history, conversation management, and message formatting"""

    def __init__(self, history_file: str = "chat_history.json"):
        self.history_file = Path(history_file)
        self.current_session: List[Dict[str, Any]] = []
        self.sessions: List[Dict[str, Any]] = []
        self._load_history()

    def add_message(
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a message to the current session

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata (sources, context, etc.)

        Returns:
            The created message
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        self.current_session.append(message)
        self._save_history()

        return message

    def add_user_message(self, content: str) -> Dict[str, Any]:
        """Add a user message to the current session"""
        return self.add_message("user", content)

    def add_assistant_message(
        self,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
        method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add an assistant message with RAG metadata

        Args:
            content: Response content
            sources: Source documents used
            context: Context provided to the model
            method: Method used (rag, direct, error)

        Returns:
            The created message
        """
        metadata = {}

        if sources:
            metadata["sources"] = [
                {
                    "content": doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content,
                    "metadata": getattr(doc, "metadata", {}),
                }
                for doc in sources
            ]

        if context:
            metadata["context_length"] = len(context)

        if method:
            metadata["method"] = method

        return self.add_message("assistant", content, metadata)

    def get_current_session(self) -> List[Dict[str, Any]]:
        """Get the current chat session"""
        return self.current_session.copy()

    def get_conversation_context(self, max_messages: int = 10) -> str:
        """
        Get recent conversation context for the LLM

        Args:
            max_messages: Maximum number of recent messages to include

        Returns:
            Formatted conversation context
        """
        if not self.current_session:
            return ""

        # Get recent messages
        recent_messages = self.current_session[-max_messages:]

        # Format as conversation
        context_parts = []
        for msg in recent_messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            context_parts.append(f"{role}: {content}")

        return "\n\n".join(context_parts)

    def start_new_session(self, title: Optional[str] = None) -> None:
        """Start a new chat session"""
        # Save current session if it has messages
        if self.current_session:
            self._save_current_session(title)

        # Reset current session
        self.current_session = []

        RichLogger.info("Started new chat session")

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all chat sessions"""
        # Include current session if it has messages
        all_sessions = self.sessions.copy()

        if self.current_session:
            all_sessions.append(
                {
                    "id": "current",
                    "title": "Current Session",
                    "created_at": self.current_session[0]["timestamp"]
                    if self.current_session
                    else datetime.now().isoformat(),
                    "message_count": len(self.current_session),
                    "messages": self.current_session,
                }
            )

        return all_sessions

    def load_session(self, session_id: str) -> bool:
        """
        Load a previous session

        Args:
            session_id: Session ID to load

        Returns:
            True if session was loaded successfully
        """
        for session in self.sessions:
            if session["id"] == session_id:
                self.current_session = session["messages"].copy()
                RichLogger.info(f"Loaded session: {session['title']}")
                return True

        RichLogger.warning(f"Session not found: {session_id}")
        return False

    def search_messages(self, query: str) -> List[Dict[str, Any]]:
        """
        Search through all messages

        Args:
            query: Search query

        Returns:
            List of matching messages with session info
        """
        results = []
        query_lower = query.lower()

        # Search in all sessions
        for session in self.get_all_sessions():
            for msg in session["messages"]:
                if query_lower in msg["content"].lower():
                    results.append(
                        {
                            "session_id": session["id"],
                            "session_title": session["title"],
                            "message": msg,
                        }
                    )

        return results

    def export_session(
        self, session_id: Optional[str] = None, format: str = "json"
    ) -> str:
        """
        Export a session or current session

        Args:
            session_id: Session ID to export (None for current)
            format: Export format (json, markdown)

        Returns:
            Exported content as string
        """
        # Get session to export
        if session_id:
            session = None
            for s in self.sessions:
                if s["id"] == session_id:
                    session = s
                    break
            if not session:
                raise ValueError(f"Session not found: {session_id}")
        else:
            # Export current session
            session = {
                "id": "current",
                "title": "Current Session",
                "created_at": self.current_session[0]["timestamp"]
                if self.current_session
                else datetime.now().isoformat(),
                "messages": self.current_session,
            }

        if format == "json":
            return json.dumps(session, indent=2)
        elif format == "markdown":
            return self._format_session_as_markdown(session)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _format_session_as_markdown(self, session: Dict[str, Any]) -> str:
        """Format a session as markdown"""
        lines = [
            f"# {session['title']}",
            f"*Created: {session['created_at']}*",
            "",
            "---",
            "",
        ]

        for msg in session["messages"]:
            role = msg["role"].capitalize()
            content = msg["content"]
            timestamp = msg["timestamp"]

            lines.append(f"### {role}")
            lines.append(f"*{timestamp}*")
            lines.append("")
            lines.append(content)

            # Add metadata if present
            if msg.get("metadata", {}).get("sources"):
                lines.append("")
                lines.append("**Sources:**")
                for i, source in enumerate(msg["metadata"]["sources"]):
                    lines.append(f"{i + 1}. {source['content']}")

            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def _save_current_session(self, title: Optional[str] = None) -> None:
        """Save current session to history"""
        if not self.current_session:
            return

        # Generate title if not provided
        if not title:
            # Use first user message as title
            for msg in self.current_session:
                if msg["role"] == "user":
                    title = (
                        msg["content"][:50] + "..."
                        if len(msg["content"]) > 50
                        else msg["content"]
                    )
                    break
            if not title:
                title = f"Session {len(self.sessions) + 1}"

        session = {
            "id": f"session_{int(datetime.now().timestamp())}",
            "title": title,
            "created_at": self.current_session[0]["timestamp"],
            "message_count": len(self.current_session),
            "messages": self.current_session,
        }

        self.sessions.append(session)
        self._save_history()

    def _load_history(self) -> None:
        """Load chat history from file"""
        if not self.history_file.exists():
            return

        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.sessions = data.get("sessions", [])
                self.current_session = data.get("current_session", [])
        except Exception as e:
            RichLogger.warning(f"Could not load chat history: {str(e)}")

    def _save_history(self) -> None:
        """Save chat history to file"""
        try:
            data = {"sessions": self.sessions, "current_session": self.current_session}

            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            RichLogger.error(f"Could not save chat history: {str(e)}")

    def clear_history(self) -> None:
        """Clear all chat history"""
        self.sessions = []
        self.current_session = []
        self._save_history()
        RichLogger.info("Chat history cleared")

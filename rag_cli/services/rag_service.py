"""Main RAG service that coordinates all operations"""

from typing import Dict, Any, Optional, List
from pathlib import Path

from rag_cli.core.rag_system import RAGSystem
from rag_cli.core.settings_manager import SettingsManager
from rag_cli.services.query_service import QueryService
from rag_cli.services.database_service import DatabaseService
from rag_cli.services.chat_service import ChatService
from rag_cli.utils.logger import RichLogger
from rag_cli.utils.defaults import (
    DEFAULT_MODEL, DEFAULT_TEMPERATURE, 
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
)


class RAGService:
    """
    Main service that coordinates RAG operations
    
    This service acts as the main entry point for all RAG-related operations,
    delegating to specialized services for specific functionality.
    """
    
    def __init__(self, settings_file: str = "settings.json"):
        # Initialize settings
        self.settings_manager = SettingsManager(settings_file)
        
        # Initialize core system
        self.rag_system = RAGSystem(settings_manager=self.settings_manager)
        
        # Initialize services
        self.query_service = QueryService(self.rag_system)
        self.database_service = DatabaseService(
            self.rag_system.vectorstore_manager,
            self.settings_manager
        )
        self.chat_service = ChatService()
        
        # Initialize system
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the RAG system with current settings"""
        try:
            # Get settings
            settings = {
                "model_name": self.settings_manager.get("model_name", DEFAULT_MODEL),
                "temperature": self.settings_manager.get("temperature", DEFAULT_TEMPERATURE),
                "chunk_size": self.settings_manager.get("chunk_size", DEFAULT_CHUNK_SIZE),
                "chunk_overlap": self.settings_manager.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
            }
            
            # Initialize database service
            self.database_service.initialize_vectorstore(
                chunk_size=settings["chunk_size"],
                chunk_overlap=settings["chunk_overlap"]
            )
            
            RichLogger.info(f"RAG Service initialized with model: {settings['model_name']}")
            
        except Exception as e:
            RichLogger.error(f"Error initializing RAG service: {str(e)}")
            raise
    
    async def process_query(
        self,
        question: str,
        use_rag: bool = True,
        temperature: Optional[float] = None,
        save_to_history: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user query
        
        Args:
            question: The user's question
            use_rag: Whether to use RAG retrieval
            temperature: Override temperature for this query
            save_to_history: Whether to save to chat history
            
        Returns:
            Response dictionary with answer and metadata
        """
        try:
            # Save user message to history
            if save_to_history:
                self.chat_service.add_user_message(question)
            
            # Process query
            result = await self.query_service.process_query(
                question=question,
                use_rag=use_rag,
                temperature=temperature
            )
            
            # Save assistant response to history
            if save_to_history:
                self.chat_service.add_assistant_message(
                    content=result["answer"],
                    sources=result.get("source_documents", []),
                    context=result.get("context"),
                    method=result.get("method")
                )
            
            return result
            
        except Exception as e:
            RichLogger.error(f"Error processing query: {str(e)}")
            error_result = {
                "answer": f"Error: {str(e)}",
                "source_documents": [],
                "context": None,
                "method": "error",
                "metadata": {"error": str(e)}
            }
            
            if save_to_history:
                self.chat_service.add_assistant_message(
                    content=error_result["answer"],
                    method="error"
                )
            
            return error_result
    
    def load_database(self, db_path: str = "./chroma_db") -> bool:
        """
        Load an existing database
        
        Args:
            db_path: Path to the database
            
        Returns:
            True if successfully loaded
        """
        success = self.database_service.load_database(db_path)
        
        if success:
            # Setup QA chain after loading database
            self.rag_system._setup_qa_chain()
        
        return success
    
    def create_database(
        self,
        docs_path: str = "./documents",
        db_path: str = "./chroma_db",
        progress_callback: Optional[Any] = None
    ) -> bool:
        """
        Create a new database from documents
        
        Args:
            docs_path: Path to documents directory
            db_path: Path where database will be created
            progress_callback: Optional progress callback
            
        Returns:
            True if successfully created
        """
        success = self.database_service.create_database(
            docs_path=docs_path,
            db_path=db_path,
            progress_callback=progress_callback
        )
        
        if success:
            # Setup QA chain after creating database
            self.rag_system._setup_qa_chain()
        
        return success
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            "settings": {
                "model": self.settings_manager.get("model_name", DEFAULT_MODEL),
                "temperature": self.settings_manager.get("temperature", DEFAULT_TEMPERATURE),
                "chunk_size": self.settings_manager.get("chunk_size", DEFAULT_CHUNK_SIZE),
                "chunk_overlap": self.settings_manager.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
                "use_query_expansion": self.settings_manager.get("use_query_expansion", False),
                "use_reranker": self.settings_manager.get("use_reranker", False)
            },
            "database": self.database_service.get_database_info(),
            "chat": {
                "current_session_messages": len(self.chat_service.current_session),
                "total_sessions": len(self.chat_service.sessions)
            },
            "system": {
                "rag_initialized": self.rag_system.qa_chain is not None,
                "vectorstore_loaded": self.rag_system.vectorstore is not None
            }
        }
    
    def update_settings(self, **kwargs) -> None:
        """
        Update system settings
        
        Args:
            **kwargs: Settings to update
        """
        # Update settings
        for key, value in kwargs.items():
            self.settings_manager.set(key, value)
        
        # Save settings
        self.settings_manager.save()
        
        # Update RAG system if needed
        if any(key in ["model_name", "temperature"] for key in kwargs):
            self.rag_system.update_settings()
        
        # Reinitialize vectorstore if chunk settings changed
        if any(key in ["chunk_size", "chunk_overlap"] for key in kwargs):
            self.database_service.initialize_vectorstore()
        
        RichLogger.info(f"Settings updated: {list(kwargs.keys())}")
    
    def reset_system(self) -> None:
        """Reset the entire system"""
        try:
            # Reset database
            self.database_service.reset_database()
            
            # Clear chat history
            self.chat_service.start_new_session()
            
            # Reset RAG system
            self.rag_system.qa_chain = None
            
            RichLogger.info("System reset successfully")
            
        except Exception as e:
            RichLogger.error(f"Error resetting system: {str(e)}")
    
    def export_chat_history(
        self,
        session_id: Optional[str] = None,
        format: str = "markdown"
    ) -> str:
        """
        Export chat history
        
        Args:
            session_id: Session to export (None for current)
            format: Export format (json, markdown)
            
        Returns:
            Exported content
        """
        return self.chat_service.export_session(session_id, format)
    
    def search_chat_history(self, query: str) -> List[Dict[str, Any]]:
        """
        Search through chat history
        
        Args:
            query: Search query
            
        Returns:
            List of matching messages
        """
        return self.chat_service.search_messages(query)
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.query_service.cleanup()
        RichLogger.info("RAG Service cleaned up")
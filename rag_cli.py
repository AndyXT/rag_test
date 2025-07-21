from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Input, RichLog, Button, Static
from textual.binding import Binding
import asyncio
import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class RAGSystem:
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm = OllamaLLM(model="llama3.2", temperature=0)
        
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
    
    def create_db_from_docs(self, docs_path="./documents", db_path="./chroma_db"):
        """Create new ChromaDB from documents"""
        loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        if not documents:
            raise ValueError(f"No documents found in {docs_path}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=db_path
        )
        self._setup_qa_chain()
        
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
            self.vectorstore.as_retriever(search_kwargs={"k": 3}), 
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

class RAGChatApp(App):
    """A Textual app for RAG chat interface."""
    
    CSS = """
    .chat-container {
        height: 1fr;
        border: solid green;
    }
    
    .input-container {
        height: 3;
        border: solid blue;
    }
    
    .status-bar {
        height: 1;
        background: $surface;
    }
    
    Input {
        width: 1fr;
    }
    
    Button {
        width: auto;
        min-width: 10;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_chat", "Clear Chat"),
        Binding("ctrl+r", "reload_db", "Reload DB"),
    ]
    
    def __init__(self):
        super().__init__()
        self.rag = RAGSystem()
        
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Static("RAG Chat - Ask questions about your documents", classes="status-bar")
        
        with Container(classes="chat-container"):
            yield RichLog(id="chat", markup=True)
        
        with Horizontal(classes="input-container"):
            yield Input(placeholder="Ask a question...", id="question_input")
            yield Button("Send", id="send_btn")
            yield Button("Load DB", id="load_btn")
            yield Button("Create DB", id="create_btn")
    
    async def on_mount(self) -> None:
        """Called when app starts."""
        chat = self.query_one("#chat", RichLog)
        chat.write("Welcome to RAG Chat!")
        chat.write("Commands: Load DB (existing) | Create DB (from ./documents) | Type questions")
        
        # Try to load existing DB automatically
        if self.rag.load_existing_db():
            chat.write("[green]✓ Loaded existing ChromaDB[/green]")
        else:
            chat.write("[yellow]No existing database found. Load or create one to start.[/yellow]")
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Called when user presses enter in input field."""
        if event.input.id == "question_input":
            await self._process_question(event.value)
            event.input.value = ""
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        chat = self.query_one("#chat", RichLog)
        
        if event.button.id == "send_btn":
            question_input = self.query_one("#question_input", Input)
            await self._process_question(question_input.value)
            question_input.value = ""
            
        elif event.button.id == "load_btn":
            chat.write("[blue]Loading existing database...[/blue]")
            if self.rag.load_existing_db():
                chat.write("[green]✓ Database loaded successfully[/green]")
            else:
                chat.write("[red]✗ No database found at ./chroma_db[/red]")
                
        elif event.button.id == "create_btn":
            chat.write("[blue]Creating database from ./documents ...[/blue]")
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.rag.create_db_from_docs
                )
                chat.write("[green]✓ Database created successfully[/green]")
            except Exception as e:
                chat.write(f"[red]✗ Error creating database: {e}[/red]")
    
    async def _process_question(self, question: str) -> None:
        """Process user question."""
        if not question.strip():
            return
            
        chat = self.query_one("#chat", RichLog)
        chat.write(f"[bold cyan]You:[/bold cyan] {question}")
        
        if not self.rag.qa_chain:
            chat.write("[red]Please load or create a database first.[/red]")
            return
        
        chat.write("[dim]Thinking...[/dim]")
        
        try:
            answer = await self.rag.query(question)
            chat.write(f"[bold green]Assistant:[/bold green] {answer}")
        except Exception as e:
            chat.write(f"[red]Error: {e}[/red]")
    
    def action_clear_chat(self) -> None:
        """Clear the chat log."""
        chat = self.query_one("#chat", RichLog)
        chat.clear()
        chat.write("Chat cleared.")
    
    def action_reload_db(self) -> None:
        """Reload the database."""
        chat = self.query_one("#chat", RichLog)
        if self.rag.load_existing_db():
            chat.write("[green]✓ Database reloaded[/green]")
        else:
            chat.write("[red]✗ No database found[/red]")

if __name__ == "__main__":
    app = RAGChatApp()
    app.run()

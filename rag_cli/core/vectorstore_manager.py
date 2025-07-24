# Vector Store Manager Module
import os
import gc
import time
import shutil
import multiprocessing as mp
from pathlib import Path
from typing import Optional, List, Dict, Any
import tempfile
from contextlib import contextmanager

# Rich imports for output formatting
from rich import print

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Reranking support
try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CrossEncoder = None
    CROSSENCODER_AVAILABLE = False


class VectorStoreManager:
    """Manages vector store operations and document processing"""

    def __init__(self, settings_manager=None):
        self.settings_manager = settings_manager
        self.embeddings = None
        self.vectorstore = None
        self.reranker = None
        self.chunk_size = 1000
        self.chunk_overlap = 200

    def initialize(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the vector store manager"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Check and increase file descriptor limit before initialization
        self._check_and_increase_fd_limit()
        
        # Initialize embeddings with better error handling
        self._initialize_embeddings_safely()
        
        # Initialize reranker if enabled
        self._initialize_reranker()

    def _check_and_increase_fd_limit(self):
        """Check and try to increase file descriptor limit"""
        try:
            import resource

            # Get current limits
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

            # Try to increase to a reasonable limit
            target_limit = min(8192, hard)

            if soft < target_limit:
                try:
                    resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, hard))
                    print(
                        f"[green]✓ Increased file descriptor limit from {soft} to {target_limit}[/green]"
                    )
                except Exception:
                    print(
                        f"[yellow]⚠ Could not increase file descriptor limit (current: {soft})[/yellow]"
                    )
                    print(
                        "[yellow]💡 Try running: ulimit -n 8192 before starting the app[/yellow]"
                    )
        except Exception:
            # Not critical if this fails
            pass

    def _ensure_cache_directories(self):
        """Ensure all required cache directories exist and are writable"""
        cache_dirs = [
            os.environ["HF_HOME"],
            os.environ["HF_DATASETS_CACHE"],
            os.path.join(os.environ["HF_HOME"], "hub"),
            os.path.join(os.environ["HF_HOME"], "transformers"),
        ]

        for cache_dir in cache_dirs:
            try:
                cache_path = Path(cache_dir)
                cache_path.mkdir(parents=True, exist_ok=True)

                # Test if directory is writable
                test_file = cache_path / ".write_test"
                test_file.touch()
                test_file.unlink()

            except (OSError, PermissionError) as e:
                print(
                    f"[yellow]⚠ Cache directory issue: {cache_dir} - {str(e)}[/yellow]"
                )
                # Try to create an alternative cache location
                alt_cache = Path.home() / ".local" / "share" / "huggingface"
                alt_cache.mkdir(parents=True, exist_ok=True)
                os.environ["HF_HOME"] = str(alt_cache)
                break

    def _clean_hf_cache_locks(self):
        """Clean up Hugging Face cache lock files that may prevent model loading"""
        cache_dir = Path(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        )

        if not cache_dir.exists():
            return

        # More aggressive patterns to clean
        lock_patterns = ["*.lock", "*.tmp*", "*incomplete*", "*.part", "*downloading*"]
        cleaned_files = []

        try:
            for pattern in lock_patterns:
                for lock_file in cache_dir.rglob(pattern):
                    try:
                        if lock_file.is_file():
                            # More aggressive: clean files older than 5 minutes
                            if time.time() - lock_file.stat().st_mtime > 300:
                                lock_file.unlink()
                                cleaned_files.append(str(lock_file))
                            # Also clean zero-byte files
                            elif lock_file.stat().st_size == 0:
                                lock_file.unlink()
                                cleaned_files.append(str(lock_file))
                    except (OSError, PermissionError):
                        continue

            # Also clean the sentence-transformers specific cache
            st_cache = cache_dir / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2"
            if st_cache.exists():
                # Clean any lock files in the model directory
                for lock_file in st_cache.rglob("*.lock"):
                    try:
                        lock_file.unlink()
                        cleaned_files.append(str(lock_file))
                    except:
                        pass

            if cleaned_files:
                print(
                    f"[green]✓ Cleaned {len(cleaned_files)} cache files (locks, temp files, etc.)[/green]"
                )

        except Exception as e:
            print(f"[yellow]⚠ Could not clean cache locks: {str(e)}[/yellow]")

    def _initialize_embeddings_safely(self):
        """Initialize embeddings with robust error handling and cache management"""
        # Ensure cache directories are properly set up
        self._ensure_cache_directories()

        # Clean any stale lock files first
        self._clean_hf_cache_locks()

        # Force single-threaded multiprocessing to avoid conflicts
        mp.set_start_method("spawn", force=True)

        try:
            # Force garbage collection before initialization
            gc.collect()

            # Get embedding model from settings or use default
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            if self.settings_manager:
                embedding_model = self.settings_manager.get("embedding_model", embedding_model)
            
            print(f"[INFO] Initializing embeddings with model: {embedding_model}")

            # Force single-threaded operation to avoid file descriptor issues
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": "cpu", "trust_remote_code": False},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 32,  # Increased for better performance
                    "convert_to_numpy": True,
                },
            )

            # Force garbage collection after initialization
            gc.collect()

        except Exception as e:
            # If initialization fails, try cache cleanup and retry once
            print(
                f"[yellow]⚠ Initial embedding initialization failed: {str(e)}[/yellow]"
            )
            print("[blue]ℹ Attempting cache cleanup and retry...[/blue]")

            self._clean_hf_cache_locks()

            # Wait a moment for file system to catch up
            time.sleep(2)

            try:
                # Retry with additional safety measures
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={
                        "device": "cpu",
                        "trust_remote_code": False,
                        "cache_dir": os.environ["HF_HOME"],  # Explicit cache directory
                    },
                    encode_kwargs={
                        "normalize_embeddings": True,
                        "batch_size": 1,
                        "convert_to_numpy": True,
                    },
                )
                print(
                    "[green]✓ Embedding initialization successful after cache cleanup[/green]"
                )

            except Exception as retry_error:
                print(
                    f"[red]✗ Failed to initialize embeddings after retry: {str(retry_error)}[/red]"
                )
                raise Exception(
                    f"Failed to initialize embeddings: {str(retry_error)}"
                ) from retry_error

    def _initialize_reranker(self):
        """Initialize reranker model if enabled"""
        if not self.settings_manager:
            return
            
        use_reranker = self.settings_manager.get("use_reranker", False)
        if not use_reranker:
            return
            
        if not CROSSENCODER_AVAILABLE:
            print("[yellow]⚠ CrossEncoder not available. Install sentence-transformers to use reranking.[/yellow]")
            return
            
        reranker_model = self.settings_manager.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        try:
            print(f"[blue]ℹ Initializing reranker model: {reranker_model}[/blue]")
            self.reranker = CrossEncoder(
                reranker_model,
                max_length=512,
                device="cpu",  # Force CPU to avoid GPU memory issues
                trust_remote_code=True  # Required for Qwen models
            )
            print("[green]✓ Reranker initialized successfully[/green]")
        except Exception as e:
            print(f"[red]✗ Failed to initialize reranker: {str(e)}[/red]")
            print("[yellow]⚠ Continuing without reranking[/yellow]")
            self.reranker = None

    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using the cross-encoder model"""
        if not self.reranker or not documents:
            return documents
            
        try:
            import asyncio
            # Get reranker settings
            reranker_top_k = self.settings_manager.get("reranker_top_k", 3) if self.settings_manager else 3
            
            # Prepare query-document pairs
            pairs = [[query, doc.page_content] for doc in documents]
            
            # Get reranking scores with timeout
            print(f"[blue]ℹ Reranking {len(documents)} documents...[/blue]")
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                future = loop.run_in_executor(None, self.reranker.predict, pairs)
                scores = loop.run_until_complete(asyncio.wait_for(future, timeout=15.0))
            except asyncio.TimeoutError:
                print(f"[red]✗ Reranking timed out after 15 seconds[/red]")
                return documents[:reranker_top_k]  # Return top k without reranking
            finally:
                loop.close()
            
            # Sort documents by score (higher is better for cross-encoder)
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top k documents
            reranked_docs = [doc for doc, score in doc_scores[:reranker_top_k]]
            
            # Print reranking results
            print(f"[green]✓ Reranked to top {len(reranked_docs)} documents[/green]")
            for i, (doc, score) in enumerate(doc_scores[:reranker_top_k]):
                preview = doc.page_content[:80].replace('\n', ' ')
                print(f"  Rank {i+1}: score={score:.3f} - {preview}...")
            
            return reranked_docs
            
        except Exception as e:
            print(f"[red]✗ Reranking failed: {str(e)}[/red]")
            print("[yellow]⚠ Using original document order[/yellow]")
            return documents

    def load_existing_db(self, db_path: str = "./chroma_db") -> bool:
        """Load existing ChromaDB with modern configuration"""
        # Clean HF cache before loading to prevent embedding issues
        print("[blue]ℹ Cleaning HuggingFace cache before loading database...[/blue]")
        self._clean_hf_cache_locks()

        # Set environment variables for ChromaDB (modern approach)
        os.environ["ANONYMIZED_TELEMETRY"] = "False"

        if os.path.exists(db_path):
            try:
                # Force garbage collection before loading
                gc.collect()

                # Use modern ChromaDB configuration
                self.vectorstore = Chroma(
                    persist_directory=db_path, embedding_function=self.embeddings
                )

                # Force garbage collection after loading
                gc.collect()

                return True
            except Exception as e:
                # If loading fails, provide more context
                print(f"[yellow]⚠ Could not load database: {str(e)}[/yellow]")
                if "fds_to_keep" in str(e):
                    print(
                        "[yellow]💡 Try restarting the application or increasing file descriptor limit[/yellow]"
                    )
                return False
        return False

    @contextmanager
    def _pdf_loader_context(self, pdf_path):
        """Context manager for PDF loading with proper cleanup"""
        loader = None
        try:
            loader = PyPDFLoader(str(pdf_path))
            yield loader
        finally:
            # Explicit cleanup
            if hasattr(loader, "close"):
                loader.close()
            del loader
            gc.collect()

    def create_db_from_docs(
        self, docs_path: str = "./documents", db_path: str = "./chroma_db", progress_callback=None
    ):
        """Create new ChromaDB from documents with robust error handling and file descriptor management"""
        # Set comprehensive environment variables to help with subprocess issues
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"

        try:
            docs_path = Path(docs_path)
            pdf_files = list(docs_path.glob("**/*.pdf"))

            if not pdf_files:
                raise ValueError(f"No PDF files found in {docs_path}")

            if progress_callback:
                progress_callback(f"Found {len(pdf_files)} PDF files...")

            # Process PDFs one by one with explicit cleanup to avoid file descriptor issues
            all_documents = []
            successful_files = []
            failed_files = []

            for i, pdf_file in enumerate(pdf_files):
                try:
                    if progress_callback:
                        progress_callback(
                            f"Processing {pdf_file.name} ({i+1}/{len(pdf_files)})..."
                        )

                    # Use context manager for proper cleanup
                    with self._pdf_loader_context(pdf_file) as loader:
                        documents = loader.load()

                        if documents:
                            all_documents.extend(documents)
                            successful_files.append(pdf_file.name)
                        else:
                            failed_files.append(f"{pdf_file.name} (no content)")

                    # Force garbage collection after each PDF to free file descriptors
                    gc.collect()

                except Exception as pdf_error:
                    failed_files.append(f"{pdf_file.name} ({str(pdf_error)[:50]}...)")
                    gc.collect()  # Clean up even on error
                    continue  # Skip problematic PDFs and continue with others

            if not all_documents:
                error_msg = "No documents could be processed successfully."
                if failed_files:
                    error_msg += f" Failed files: {', '.join(failed_files[:3])}"
                    if len(failed_files) > 3:
                        error_msg += f" and {len(failed_files) - 3} more"
                raise ValueError(error_msg)

            if progress_callback:
                msg = f"Successfully loaded {len(successful_files)} files"
                if failed_files:
                    msg += f" ({len(failed_files)} failed)"
                progress_callback(msg)

            if progress_callback:
                progress_callback(f"Splitting {len(all_documents)} documents...")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            texts = text_splitter.split_documents(all_documents)

            # Clear large document list to free memory
            del all_documents
            gc.collect()

            if progress_callback:
                progress_callback(f"Creating embeddings for {len(texts)} chunks...")

            # Create vectorstore with robust file descriptor handling
            self.vectorstore = self._create_chroma_db(texts, db_path)

            success_msg = f"Database created with {len(texts)} chunks from {len(successful_files)} files"
            if failed_files:
                success_msg += f" ({len(failed_files)} files skipped due to errors)"

            if progress_callback:
                progress_callback(success_msg)

        except Exception as e:
            # Force cleanup on error
            gc.collect()
            # Re-raise with more context
            raise Exception(f"Database creation failed: {str(e)}") from e

    def _create_chroma_db(self, texts: List[Document], db_path: str) -> Chroma:
        """Create ChromaDB with modern configuration and robust file descriptor handling"""
        # Set comprehensive environment variables for ChromaDB
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        os.environ["CHROMA_SERVER_NOFILE"] = (
            "65536"  # Increase file descriptor limit for ChromaDB
        )

        final_path = Path(db_path)
        backup_path = None
        temp_path = None

        try:
            # Clean up old backups first
            self._cleanup_old_backups(db_path)

            # Create backup of existing database if it exists
            if final_path.exists():
                backup_path = Path(str(final_path) + f"_backup_{int(time.time())}")
                print(f"[blue]ℹ Backing up existing database to {backup_path}[/blue]")
                shutil.move(str(final_path), str(backup_path))

            # Create temporary directory for new database
            temp_dir = tempfile.mkdtemp(prefix="chroma_temp_")
            temp_path = Path(temp_dir)

            # Create ChromaDB with modern configuration
            # Use smaller batches and add delays to avoid file descriptor exhaustion
            batch_size = 100  # Increased for better performance with Qwen models
            delay_between_batches = 0.05  # Reduced delay

            vectorstore = None
            total_batches = (len(texts) + batch_size - 1) // batch_size

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_num = (i // batch_size) + 1

                try:
                    if vectorstore is None:
                        # Create initial vectorstore with explicit settings in temp directory
                        vectorstore = Chroma.from_documents(
                            documents=batch,
                            embedding=self.embeddings,
                            persist_directory=str(temp_path),
                            collection_metadata={"hnsw:space": "cosine"},
                        )
                    else:
                        # Add to existing vectorstore
                        vectorstore.add_documents(batch)

                    # Force garbage collection and small delay between batches
                    gc.collect()
                    if batch_num < total_batches:  # Don't delay after last batch
                        time.sleep(delay_between_batches)

                except Exception:
                    # If a batch fails, try to continue with smaller batches
                    if batch_size > 5:
                        # Retry with smaller batch size
                        smaller_batch_size = max(5, batch_size // 2)
                        for j in range(
                            i, min(i + batch_size, len(texts)), smaller_batch_size
                        ):
                            small_batch = texts[j : j + smaller_batch_size]
                            try:
                                if vectorstore is None:
                                    vectorstore = Chroma.from_documents(
                                        documents=small_batch,
                                        embedding=self.embeddings,
                                        persist_directory=db_path,
                                        collection_metadata={"hnsw:space": "cosine"},
                                    )
                                else:
                                    vectorstore.add_documents(small_batch)
                                gc.collect()
                                time.sleep(delay_between_batches)
                            except Exception:
                                # Skip this small batch and continue
                                continue
                    else:
                        # Skip this batch entirely if we can't make it smaller
                        continue

            if vectorstore is None:
                raise Exception(
                    "Failed to create any vectorstore - all batches failed. This may be due to embedding model issues or document processing problems."
                )

            # If we get here, database creation was successful
            # Verify the temporary database is valid before moving
            if temp_path and temp_path.exists():
                # Quick validation - check if essential files exist
                essential_files = ["chroma.sqlite3"]
                temp_db_valid = all((temp_path / f).exists() for f in essential_files)

                if temp_db_valid:
                    shutil.move(str(temp_path), str(final_path))
                    print(
                        f"[green]✓ Database successfully created at {final_path}[/green]"
                    )

                    # Remove backup if creation was successful
                    if backup_path and backup_path.exists():
                        try:
                            shutil.rmtree(str(backup_path))
                            print("[green]✓ Removed backup database[/green]")
                        except Exception:
                            print(
                                f"[yellow]⚠ Could not remove backup at {backup_path}[/yellow]"
                            )
                else:
                    raise Exception(
                        "Created database appears to be incomplete or corrupted"
                    )

            # Final garbage collection
            gc.collect()
            return vectorstore

        except Exception as e:
            # Clean up temporary directory on failure
            if temp_path and temp_path.exists():
                try:
                    shutil.rmtree(str(temp_path))
                    print("[blue]ℹ Cleaned up temporary database[/blue]")
                except Exception:
                    pass

            # Restore backup if it exists
            if backup_path and backup_path.exists():
                try:
                    shutil.move(str(backup_path), str(final_path))
                    print("[green]✓ Restored backup database[/green]")
                except Exception as restore_error:
                    print(f"[red]✗ Failed to restore backup: {restore_error}[/red]")

            gc.collect()
            raise Exception(f"ChromaDB creation failed: {str(e)}") from e

    def _cleanup_old_backups(self, db_path: str, max_backups: int = 3):
        """Clean up old backup directories to prevent disk space issues"""
        try:
            base_path = Path(db_path)
            parent_dir = base_path.parent
            backup_pattern = f"{base_path.name}_backup_*"

            # Find all backup directories
            backup_dirs = []
            for item in parent_dir.glob(backup_pattern):
                if item.is_dir():
                    try:
                        # Extract timestamp from backup name
                        timestamp = int(item.name.split("_backup_")[1])
                        backup_dirs.append((timestamp, item))
                    except (ValueError, IndexError):
                        continue

            # Sort by timestamp (newest first) and remove old backups
            backup_dirs.sort(reverse=True)
            for _, backup_dir in backup_dirs[max_backups:]:
                try:
                    shutil.rmtree(str(backup_dir))
                    print(f"[blue]ℹ Removed old backup: {backup_dir.name}[/blue]")
                except Exception as e:
                    print(
                        f"[yellow]⚠ Could not remove old backup {backup_dir}: {e}[/yellow]"
                    )

        except Exception as e:
            print(f"[yellow]⚠ Error during backup cleanup: {e}[/yellow]")

    def get_vectorstore(self) -> Optional[Chroma]:
        """Get the current vectorstore instance"""
        return self.vectorstore

    def get_embeddings(self) -> Optional[HuggingFaceEmbeddings]:
        """Get the embeddings instance"""
        return self.embeddings

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get database statistics"""
        if not self.vectorstore:
            return None

        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "document_count": count,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }
        except Exception:
            return {"document_count": "Unknown"}
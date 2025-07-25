# LLM Manager Module
import os
from typing import Optional, Any, List
from pathlib import Path

# Import our logger and defaults
from rag_cli.utils.logger import RichLogger
from rag_cli.utils.defaults import (
    DEFAULT_OLLAMA_MODEL, DEFAULT_TEMPERATURE, LARGE_MODELS,
    DEFAULT_QUERY_EXPANSION_MODEL
)

# LangChain imports
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic  # type: ignore
from langchain_core.language_models.base import BaseLanguageModel


class LLMManager:
    """Manages LLM initialization and operations"""

    def __init__(self, settings_manager=None):
        self.settings_manager = settings_manager
        self.llm: Optional[BaseLanguageModel] = None
        self.query_expansion_llm: Optional[BaseLanguageModel] = None
        self.model_name = DEFAULT_OLLAMA_MODEL
        self.temperature = 0.1

    def initialize(self, model_name: str = DEFAULT_OLLAMA_MODEL, temperature: float = DEFAULT_TEMPERATURE):
        """Initialize the main LLM"""
        self.model_name = model_name
        self.temperature = temperature
        self._initialize_llm()
        self._initialize_query_expansion_llm()

    def _initialize_llm(self):
        """Initialize LLM based on provider settings"""
        if not self.settings_manager:
            # Fallback to Ollama if no settings manager
            self._initialize_ollama()
            return

        provider = self.settings_manager.get("llm_provider", "ollama")

        try:
            if provider == "ollama":
                self._initialize_ollama()
            elif provider == "openai":
                self._initialize_openai()
            elif provider == "anthropic":
                self._initialize_anthropic()
            else:
                RichLogger.warning(f"Unknown provider '{provider}', falling back to Ollama")
                self._initialize_ollama()
        except Exception as e:
            RichLogger.error(f"Failed to initialize {provider}: {str(e)}")
            if provider != "ollama":
                RichLogger.warning("Falling back to Ollama")
                try:
                    self._initialize_ollama()
                except Exception as fallback_error:
                    RichLogger.error(f"Ollama fallback also failed: {str(fallback_error)}")
                    RichLogger.error("Please ensure Ollama is installed and running, or configure an API provider")
                    raise fallback_error
            else:
                raise e

    def _initialize_ollama(self):
        """Initialize Ollama LLM"""
        # Get the Ollama model from settings
        ollama_model = self.settings_manager.get("ollama_model", DEFAULT_OLLAMA_MODEL) if self.settings_manager else self.model_name
        
        try:
            self.llm = OllamaLLM(
                model=ollama_model,
                temperature=self.temperature,
                num_ctx=2048,  # Reduce context window to save memory
                num_thread=1,  # Use single thread to avoid fd issues
                keep_alive="5m",  # Keep model loaded for only 5 minutes to free memory faster
            )
            RichLogger.success(f"Initialized Ollama with model: {ollama_model}")
            
            # Check if this is a large model that might conflict with query expansion
            large_models = LARGE_MODELS[:4]  # Use first 4 large models
            if any(ollama_model.startswith(m) for m in large_models):
                RichLogger.warning(f"Large model detected ({ollama_model}). Query expansion may cause memory issues.")
                if self.settings_manager and self.settings_manager.get("use_query_expansion", False):
                    RichLogger.warning("Consider disabling query expansion for better performance.")
                    
        except Exception:
            # Fallback to simpler initialization
            self.llm = OllamaLLM(model=ollama_model, temperature=self.temperature)
            RichLogger.success(f"Initialized Ollama (simple mode) with model: {ollama_model}")

    def _initialize_openai(self):
        """Initialize OpenAI LLM"""
        # Check environment variable first, fall back to settings
        api_key = os.environ.get("OPENAI_API_KEY") or self.settings_manager.get("api_key", "")
        api_base = os.environ.get("OPENAI_API_BASE") or self.settings_manager.get("api_base_url", "")
        model = self.settings_manager.get("openai_model", "gpt-3.5-turbo")

        if not api_key or not api_key.strip():
            raise ValueError("OpenAI API key is required but is missing or contains only whitespace. Set OPENAI_API_KEY environment variable or provide in settings.")

        kwargs = {"model": model, "temperature": self.temperature, "api_key": api_key}

        if api_base:
            kwargs["base_url"] = api_base

        self.llm = ChatOpenAI(**kwargs)
        api_source = "environment" if os.environ.get("OPENAI_API_KEY") else "settings"
        RichLogger.success(f"Initialized OpenAI with model: {model} (API key from {api_source})")

    def _initialize_anthropic(self):
        """Initialize Anthropic LLM"""
        # Check environment variable first, fall back to settings
        api_key = os.environ.get("ANTHROPIC_API_KEY") or self.settings_manager.get("api_key", "")
        model = self.settings_manager.get("anthropic_model", "claude-3-haiku-20240307")

        if not api_key or not api_key.strip():
            raise ValueError(
                "Anthropic API key is required but not provided. Set ANTHROPIC_API_KEY environment variable or provide in settings."
            )

        self.llm = ChatAnthropic(
            model=model, temperature=self.temperature, api_key=api_key
        )
        api_source = "environment" if os.environ.get("ANTHROPIC_API_KEY") else "settings"
        RichLogger.success(f"Initialized Anthropic with model: {model} (API key from {api_source})")

    def _initialize_query_expansion_llm(self):
        """Initialize query expansion LLM if enabled"""
        if not self.settings_manager:
            return
            
        use_query_expansion = self.settings_manager.get("use_query_expansion", False)
        if not use_query_expansion:
            return
            
        # Check if main model is large - if so, skip query expansion
        main_model = self.settings_manager.get("ollama_model", DEFAULT_OLLAMA_MODEL)
        large_models = LARGE_MODELS
        if any(main_model.startswith(m) for m in large_models):
            RichLogger.warning(f"Large main model detected ({main_model}). Skipping query expansion to prevent memory issues.")
            RichLogger.info("To use query expansion, switch to a smaller main model (e.g., qwen2.5-coder:7b)")
            self.query_expansion_llm = None
            return
            
        query_expansion_model = self.settings_manager.get("query_expansion_model", DEFAULT_QUERY_EXPANSION_MODEL)
        
        try:
            RichLogger.info(f"Initializing query expansion LLM: {query_expansion_model}")
            # Use OllamaLLM for query expansion with memory-efficient settings
            self.query_expansion_llm = OllamaLLM(
                model=query_expansion_model,
                temperature=0.3,  # Low temperature for consistent expansions
                num_ctx=512,  # Small context window for query expansion
                keep_alive="1m",  # Unload quickly after use
            )
            RichLogger.success("Query expansion LLM initialized successfully")
        except Exception as e:
            RichLogger.error(f"Failed to initialize query expansion LLM: {str(e)}")
            RichLogger.warning("Continuing without query expansion")
            self.query_expansion_llm = None

    def expand_query(self, original_query: str, expansion_count: int = 3) -> List[str]:
        """Expand a query using the small LLM to improve retrieval"""
        if not self.query_expansion_llm:
            return [original_query]
            
        try:
            import asyncio
            
            # Create prompt for query expansion
            expansion_prompt = f"""Given this question about Rust programming: "{original_query}"

Generate {expansion_count} alternative phrasings or expanded versions that would help find relevant information. Include:
1. A rephrased version using different terminology
2. A version with added context or related terms
3. A more specific or technical version

Format each query on a new line. Only output the queries, no explanations.

Queries:"""
            
            RichLogger.info(f"Expanding query: {original_query}")
            
            # Get expanded queries with timeout protection
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                future = loop.run_in_executor(None, self.query_expansion_llm.invoke, expansion_prompt)
                response = loop.run_until_complete(asyncio.wait_for(future, timeout=10.0))
            except asyncio.TimeoutError:
                RichLogger.error("Query expansion timed out after 10 seconds")
                return [original_query]
            finally:
                loop.close()
            
            # Parse response into individual queries
            expanded_queries = [q.strip() for q in response.split('\n') if q.strip()]
            
            # Always include original query
            all_queries = [original_query] + expanded_queries[:expansion_count]
            
            RichLogger.success(f"Generated {len(all_queries)} query variations")
            for i, q in enumerate(all_queries):
                RichLogger.debug(f"  Query {i+1}: {q[:80]}...")
            
            return all_queries
            
        except Exception as e:
            RichLogger.error(f"Query expansion failed: {str(e)}")
            return [original_query]

    def get_current_model_name(self) -> str:
        """Get the current model name based on the active provider"""
        if not self.settings_manager:
            return self.model_name
            
        provider = self.settings_manager.get("llm_provider", "ollama")
        
        if provider == "ollama":
            return self.settings_manager.get("ollama_model", DEFAULT_OLLAMA_MODEL)
        elif provider == "openai":
            return self.settings_manager.get("openai_model", "gpt-3.5-turbo")
        elif provider == "anthropic":
            return self.settings_manager.get("anthropic_model", "claude-3-haiku-20240307")
        else:
            return self.model_name  # Fallback

    def get_llm(self) -> Optional[BaseLanguageModel]:
        """Get the initialized LLM instance"""
        return self.llm

    def get_query_expansion_llm(self) -> Optional[BaseLanguageModel]:
        """Get the query expansion LLM instance"""
        return self.query_expansion_llm

    def invoke(self, prompt: str) -> str:
        """Invoke the LLM with a prompt"""
        if not self.llm:
            raise ValueError("LLM not initialized")
        
        response = self.llm.invoke(prompt)
        
        # Convert response to string if it's an object
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
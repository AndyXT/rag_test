"""LLM Manager Module.

This module manages Large Language Model (LLM) interactions for the RAG application.
It supports multiple LLM providers including Ollama (local), OpenAI, and Anthropic,
handles model initialization, query expansion, and provides a unified interface for
LLM operations.
"""
import os
from typing import Optional, List, Any, Dict

# Import our logger and defaults
from rag_cli.utils.logger import RichLogger
from rag_cli.utils.defaults import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TEMPERATURE,
    LARGE_MODELS,
    DEFAULT_QUERY_EXPANSION_MODEL,
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

    def initialize(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
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
            elif provider == "unsloth":
                self._initialize_unsloth()
            elif provider == "mlx":
                self._initialize_mlx()
            else:
                RichLogger.warning(
                    f"Unknown provider '{provider}', falling back to Ollama"
                )
                self._initialize_ollama()
        except Exception as e:
            RichLogger.error(f"Failed to initialize {provider}: {str(e)}")
            if provider != "ollama":
                RichLogger.warning("Falling back to Ollama")
                try:
                    self._initialize_ollama()
                except Exception as fallback_error:
                    RichLogger.error(
                        f"Ollama fallback also failed: {str(fallback_error)}"
                    )
                    RichLogger.error(
                        "Please ensure Ollama is installed and running, or configure an API provider"
                    )
                    raise fallback_error
            else:
                raise e

    def _initialize_ollama(self):
        """Initialize Ollama LLM"""
        # Get the Ollama model from settings
        ollama_model = (
            self.settings_manager.get("ollama_model", DEFAULT_OLLAMA_MODEL)
            if self.settings_manager
            else self.model_name
        )

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
                RichLogger.warning(
                    f"Large model detected ({ollama_model}). Query expansion may cause memory issues."
                )
                if self.settings_manager and self.settings_manager.get(
                    "use_query_expansion", False
                ):
                    RichLogger.warning(
                        "Consider disabling query expansion for better performance."
                    )

        except Exception:
            # Fallback to simpler initialization
            self.llm = OllamaLLM(model=ollama_model, temperature=self.temperature)
            RichLogger.success(
                f"Initialized Ollama (simple mode) with model: {ollama_model}"
            )

    def _initialize_openai(self):
        """Initialize OpenAI LLM"""
        # Check environment variable first, fall back to settings
        api_key = os.environ.get("OPENAI_API_KEY") or self.settings_manager.get(
            "api_key", ""
        )
        api_base = os.environ.get("OPENAI_API_BASE") or self.settings_manager.get(
            "api_base_url", ""
        )
        model = self.settings_manager.get("openai_model", "gpt-3.5-turbo")

        if not api_key or not api_key.strip():
            raise ValueError(
                "OpenAI API key is required but is missing or contains only whitespace. "
                "Set OPENAI_API_KEY environment variable or provide in settings."
            )

        kwargs = {"model": model, "temperature": self.temperature, "api_key": api_key}

        if api_base:
            kwargs["base_url"] = api_base

        self.llm = ChatOpenAI(**kwargs)
        api_source = "environment" if os.environ.get("OPENAI_API_KEY") else "settings"
        RichLogger.success(
            f"Initialized OpenAI with model: {model} (API key from {api_source})"
        )

    def _initialize_anthropic(self):
        """Initialize Anthropic LLM"""
        # Check environment variable first, fall back to settings
        api_key = os.environ.get("ANTHROPIC_API_KEY") or self.settings_manager.get(
            "api_key", ""
        )
        model = self.settings_manager.get("anthropic_model", "claude-3-haiku-20240307")

        if not api_key or not api_key.strip():
            raise ValueError(
                "Anthropic API key is required but not provided. "
                "Set ANTHROPIC_API_KEY environment variable or provide in settings."
            )

        self.llm = ChatAnthropic(
            model=model, temperature=self.temperature, api_key=api_key
        )
        api_source = (
            "environment" if os.environ.get("ANTHROPIC_API_KEY") else "settings"
        )
        RichLogger.success(
            f"Initialized Anthropic with model: {model} (API key from {api_source})"
        )

    def _initialize_unsloth(self):
        """Initialize Unsloth LLM"""
        # Get model configuration from settings
        unsloth_model = self.settings_manager.get(
            "unsloth_model", "unsloth/Qwen2.5-Coder-14B-Instruct"
        )
        max_seq_length = self.settings_manager.get("unsloth_max_seq_length", 8192)
        load_in_4bit = self.settings_manager.get("unsloth_4bit", True)
        load_in_8bit = self.settings_manager.get("unsloth_8bit", False)
        
        # Check for CUDA availability
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Unsloth requires CUDA-capable GPU. No CUDA device found."
                )
        except ImportError:
            raise ImportError(
                "PyTorch is required for Unsloth. Install it with: pip install torch"
            )
        
        try:
            self.llm = UnslothLLM(
                model_name=unsloth_model,
                temperature=self.temperature,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
            )
            
            RichLogger.success(f"Initialized Unsloth with model: {unsloth_model}")
            
            # Log quantization info
            if load_in_4bit:
                RichLogger.info("Using 4-bit quantization for reduced memory usage")
            elif load_in_8bit:
                RichLogger.info("Using 8-bit quantization for better accuracy")
            else:
                RichLogger.info("Using full precision (no quantization)")
                
        except Exception as e:
            RichLogger.error(f"Failed to initialize Unsloth: {str(e)}")
            raise

    def _initialize_mlx(self):
        """Initialize MLX LLM for Apple Silicon"""
        # Check platform
        import platform
        if platform.system() != "Darwin":
            raise RuntimeError(
                "MLX is only supported on macOS. Current platform: " + platform.system()
            )
        
        # Get model configuration from settings
        mlx_model_path = self.settings_manager.get(
            "mlx_model_path", 
            os.path.expanduser("~/.cache/mlx_models/mistral-7b-instruct")
        )
        max_tokens = self.settings_manager.get("mlx_max_tokens", 2048)
        seed = self.settings_manager.get("mlx_seed", 42)
        
        # Check if model path exists
        if not os.path.exists(mlx_model_path):
            raise FileNotFoundError(
                f"MLX model not found at: {mlx_model_path}\n"
                f"Please download a model first. You can:\n"
                f"1. Use Hugging Face MLX models: huggingface-cli download mlx-community/Mistral-7B-Instruct-v0.2-4bit\n"
                f"2. Convert your own model using mlx_lm.convert"
            )
        
        # Check required files exist
        required_files = ["config.json", "tokenizer.model"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(mlx_model_path, f))]
        if missing_files:
            raise FileNotFoundError(
                f"Missing required files in {mlx_model_path}: {', '.join(missing_files)}"
            )
        
        try:
            self.llm = MLXLLM(
                model_path=mlx_model_path,
                temperature=self.temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            
            RichLogger.success(f"Initialized MLX with model at: {mlx_model_path}")
            RichLogger.info(f"Using temperature: {self.temperature}, max_tokens: {max_tokens}")
            
        except Exception as e:
            RichLogger.error(f"Failed to initialize MLX: {str(e)}")
            raise

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
            RichLogger.warning(
                f"Large main model detected ({main_model}). Skipping query expansion to prevent memory issues."
            )
            RichLogger.info(
                "To use query expansion, switch to a smaller main model (e.g., qwen2.5-coder:7b)"
            )
            self.query_expansion_llm = None
            return

        query_expansion_model = self.settings_manager.get(
            "query_expansion_model", DEFAULT_QUERY_EXPANSION_MODEL
        )

        try:
            RichLogger.info(
                f"Initializing query expansion LLM: {query_expansion_model}"
            )
            # Use OllamaLLM for query expansion with memory-efficient settings
            self.query_expansion_llm = OllamaLLM(
                model=query_expansion_model,
                temperature=0.3,  # Low temperature for consistent expansions
                num_ctx=512,  # Small context window for query expansion
                keep_alive="0s",  # Unload immediately after use to free memory
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

Generate {expansion_count} alternative phrasings or expanded versions that would help find relevant 
information. Include:
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
                future = loop.run_in_executor(
                    None, self.query_expansion_llm.invoke, expansion_prompt
                )
                response = loop.run_until_complete(
                    asyncio.wait_for(future, timeout=10.0)
                )
            except asyncio.TimeoutError:
                RichLogger.error("Query expansion timed out after 10 seconds")
                return [original_query]
            finally:
                loop.close()

            # Parse response into individual queries
            expanded_queries = [q.strip() for q in response.split("\n") if q.strip()]

            # Always include original query
            all_queries = [original_query] + expanded_queries[:expansion_count]

            RichLogger.success(f"Generated {len(all_queries)} query variations")
            for i, q in enumerate(all_queries):
                RichLogger.debug(f"  Query {i + 1}: {q[:80]}...")

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
            return self.settings_manager.get(
                "anthropic_model", "claude-3-haiku-20240307"
            )
        elif provider == "unsloth":
            return self.settings_manager.get(
                "unsloth_model", "unsloth/Qwen2.5-Coder-14B-Instruct"
            )
        elif provider == "mlx":
            mlx_path = self.settings_manager.get(
                "mlx_model_path", "~/.cache/mlx_models/mistral-7b-instruct"
            )
            return os.path.basename(mlx_path)
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
        if hasattr(response, "content"):
            return response.content
        else:
            return str(response)

class UnslothLLM(BaseLanguageModel):
    """Custom LangChain wrapper for Unsloth models"""
    
    def __init__(
        self,
        model_name: str = "unsloth/Qwen2.5-Coder-14B-Instruct",
        max_seq_length: int = 8192,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        temperature: float = 0.1,
        max_new_tokens: int = 2048,
        **kwargs
    ):
        """Initialize Unsloth model wrapper
        
        Args:
            model_name: Unsloth model name from HuggingFace
            max_seq_length: Maximum sequence length
            load_in_4bit: Use 4-bit quantization
            load_in_8bit: Use 8-bit quantization
            temperature: Generation temperature
            max_new_tokens: Maximum tokens to generate
        """
        super().__init__()
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Unsloth model and tokenizer"""
        try:
            from unsloth import FastLanguageModel
            
            RichLogger.info(f"Loading Unsloth model: {self.model_name}")
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                dtype=None,  # Auto-detect
                device_map="auto",
            )
            
            # Set model to eval mode for inference
            FastLanguageModel.for_inference(self.model)
            
            RichLogger.success(f"Unsloth model loaded successfully: {self.model_name}")
            
        except ImportError:
            raise ImportError(
                "Unsloth is not installed. Install it with: pip install unsloth"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Unsloth model: {str(e)}")
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from prompt"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length,
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode and extract only the new tokens
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from response
            if full_response.startswith(prompt):
                response = full_response[len(prompt):].strip()
            else:
                response = full_response.strip()
            
            return response
            
        except Exception as e:
            RichLogger.error(f"Unsloth generation error: {str(e)}")
            raise
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Async generation - just calls sync version"""
        return self._call(prompt, stop, run_manager, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM"""
        return "unsloth"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return model parameters for identification"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_seq_length": self.max_seq_length,
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
        }
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke the model - implements BaseLanguageModel interface"""
        return self._call(prompt, **kwargs)

class MLXLLM(BaseLanguageModel):
    """Custom LangChain wrapper for MLX models on Apple Silicon"""
    
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        seed: int = 42,
        **kwargs
    ):
        """Initialize MLX model wrapper
        
        Args:
            model_path: Path to MLX model directory containing weights.npz, config.json, and tokenizer.model
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.model = None
        self.tokenizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the MLX model and tokenizer"""
        try:
            import platform
            if platform.system() != "Darwin" or platform.processor() != "arm":
                raise RuntimeError(
                    "MLX requires Apple Silicon Mac. Detected non-Apple Silicon system."
                )
            
            import mlx
            import mlx.core as mx
            from mlx_lm import load, generate
            
            RichLogger.info(f"Loading MLX model from: {self.model_path}")
            
            # Load model and tokenizer using mlx-lm
            self.model, self.tokenizer = load(self.model_path)
            
            # Set random seed
            mx.random.seed(self.seed)
            
            RichLogger.success(f"MLX model loaded successfully from: {self.model_path}")
            
        except ImportError:
            raise ImportError(
                "MLX is not installed. Install it with: pip install mlx mlx-lm"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MLX model: {str(e)}")
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from prompt"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized")
        
        try:
            from mlx_lm import generate
            
            # Generate response using mlx-lm
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                temp=self.temperature,
                verbose=False,  # Disable verbose output
            )
            
            # Extract only the generated text (remove the prompt)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            RichLogger.error(f"MLX generation error: {str(e)}")
            raise
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Async generation - just calls sync version"""
        return self._call(prompt, stop, run_manager, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM"""
        return "mlx"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return model parameters for identification"""
        return {
            "model_path": self.model_path,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke the model - implements BaseLanguageModel interface"""
        return self._call(prompt, **kwargs)

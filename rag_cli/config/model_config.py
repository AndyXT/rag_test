"""Model-related configuration constants"""

# LLM Model defaults by provider
OLLAMA_MODELS = {
    "default": "llama3.2:3b",
    "small": "llama3.2:3b",
    "medium": "llama3.2:7b",
    "large": "llama3.1:70b",
    "coding": "qwen2.5-coder:7b",
    "coding_large": "qwen2.5-coder:32b",
}

OPENAI_MODELS = {
    "default": "gpt-3.5-turbo",
    "small": "gpt-3.5-turbo",
    "medium": "gpt-4",
    "large": "gpt-4-turbo",
}

ANTHROPIC_MODELS = {
    "default": "claude-3-haiku-20240307",
    "small": "claude-3-haiku-20240307",
    "medium": "claude-3-sonnet-20240229",
    "large": "claude-3-opus-20240229",
}

# Unsloth models (GPU-accelerated quantized models)
UNSLOTH_MODELS = {
    "default": "unsloth/Qwen2.5-Coder-14B-Instruct",
    "small": "unsloth/Llama-3.2-3B",
    "medium": "unsloth/mistral-7b-instruct-v0.3",
    "large": "unsloth/Qwen2.5-Coder-14B-Instruct",
    "phi": "unsloth/Phi-4",
    "llama-8b": "unsloth/Llama-3.1-8B",
    "gemma-4b": "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "gemma-12b": "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "gemma-27b": "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
    "qwen-4b": "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    "qwen-8b": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
}

# MLX models (Apple Silicon optimized)
MLX_MODELS = {
    "default": "mistral-7b-instruct",
    "mistral-7b": "mistral-7b-instruct",
    "mistral-7b-4bit": "Mistral-7B-Instruct-v0.2-4bit",
    "llama-3b": "Llama-3.2-3B-Instruct-4bit",
    "phi-3": "Phi-3-mini-4k-instruct-4bit",
    "qwen-7b": "Qwen2.5-7B-Instruct-4bit",
    "gemma-2b": "gemma-2b-it-4bit",
    "starcoder-1b": "starcoder-1b-4bit",
}

# Default model (for backward compatibility)
DEFAULT_MODEL = OLLAMA_MODELS["default"]
DEFAULT_OLLAMA_MODEL = OLLAMA_MODELS["default"]
DEFAULT_OPENAI_MODEL = OPENAI_MODELS["default"]
DEFAULT_ANTHROPIC_MODEL = ANTHROPIC_MODELS["default"]
DEFAULT_UNSLOTH_MODEL = UNSLOTH_MODELS["default"]
DEFAULT_MLX_MODEL = MLX_MODELS["default"]

# Embedding models
EMBEDDING_MODELS = {
    "default": "sentence-transformers/all-MiniLM-L6-v2",
    "small": "sentence-transformers/all-MiniLM-L6-v2",
    "medium": "sentence-transformers/all-mpnet-base-v2",
    "large": "sentence-transformers/all-roberta-large-v1",
    "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}

DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODELS["default"]

# Reranker models
RERANKER_MODELS = {
    "default": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "small": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "medium": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "large": "cross-encoder/ms-marco-base-v2",
    "multilingual": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
}

DEFAULT_RERANKER_MODEL = RERANKER_MODELS["default"]

# Query expansion model
DEFAULT_QUERY_EXPANSION_MODEL = OLLAMA_MODELS["small"]

# Model parameters
MODEL_PARAMETERS = {
    "temperature": {"default": 0.1, "creative": 0.7, "factual": 0.0, "balanced": 0.3},
    "timeout": {"default": 30, "quick": 10, "normal": 30, "extended": 60, "long": 120},
    "max_tokens": {
        "default": None,  # Use model default
        "short": 256,
        "medium": 512,
        "long": 1024,
        "very_long": 2048,
    },
}

DEFAULT_TEMPERATURE = MODEL_PARAMETERS["temperature"]["default"]
DEFAULT_TIMEOUT = MODEL_PARAMETERS["timeout"]["default"]

# Large models that require more resources
LARGE_MODELS = [
    "qwen2.5-coder:32b",
    "mixtral:8x7b",
    "llama3.1:70b",
    "qwen2.5:32b",
    "deepseek-coder:33b",
    "yi:34b",
    "solar:10.7b",
]

# Model capabilities
MODEL_CAPABILITIES = {
    "supports_tools": [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "claude-3-opus",
        "claude-3-sonnet",
        "llama3.1:70b",
        "llama3.1:8b",
    ],
    "supports_vision": [
        "gpt-4-vision",
        "gpt-4-turbo",
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
        "llava",
        "bakllava",
    ],
    "supports_long_context": [
        "claude-3-opus",
        "claude-3-sonnet",
        "gpt-4-turbo",
        "gpt-4-32k",
        "yi:34b",
        "qwen2.5:32b",
        "unsloth/Qwen2.5-Coder-14B-Instruct",
        "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    ],
    "supports_gpu_acceleration": [
        "unsloth/Qwen2.5-Coder-14B-Instruct",
        "unsloth/Llama-3.1-8B",
        "unsloth/Llama-3.2-3B",
        "unsloth/mistral-7b-instruct-v0.3",
        "unsloth/Phi-4",
        "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
        "unsloth/Qwen3-4B-unsloth-bnb-4bit",
        "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    ],
    "supports_apple_silicon": [
        "mistral-7b-instruct",
        "Mistral-7B-Instruct-v0.2-4bit",
        "Llama-3.2-3B-Instruct-4bit",
        "Phi-3-mini-4k-instruct-4bit",
        "Qwen2.5-7B-Instruct-4bit",
        "gemma-2b-it-4bit",
        "starcoder-1b-4bit",
    ],
}

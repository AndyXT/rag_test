"""Model-related configuration constants"""

# LLM Model defaults by provider
OLLAMA_MODELS = {
    "default": "llama3.2:3b",
    "small": "llama3.2:3b",
    "medium": "llama3.2:7b",
    "large": "llama3.1:70b",
    "coding": "qwen2.5-coder:7b",
    "coding_large": "qwen2.5-coder:32b"
}

OPENAI_MODELS = {
    "default": "gpt-3.5-turbo",
    "small": "gpt-3.5-turbo",
    "medium": "gpt-4",
    "large": "gpt-4-turbo"
}

ANTHROPIC_MODELS = {
    "default": "claude-3-haiku-20240307",
    "small": "claude-3-haiku-20240307",
    "medium": "claude-3-sonnet-20240229",
    "large": "claude-3-opus-20240229"
}

# Default model (for backward compatibility)
DEFAULT_MODEL = OLLAMA_MODELS["default"]
DEFAULT_OLLAMA_MODEL = OLLAMA_MODELS["default"]
DEFAULT_OPENAI_MODEL = OPENAI_MODELS["default"]
DEFAULT_ANTHROPIC_MODEL = ANTHROPIC_MODELS["default"]

# Embedding models
EMBEDDING_MODELS = {
    "default": "sentence-transformers/all-MiniLM-L6-v2",
    "small": "sentence-transformers/all-MiniLM-L6-v2",
    "medium": "sentence-transformers/all-mpnet-base-v2",
    "large": "sentence-transformers/all-roberta-large-v1",
    "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}

DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODELS["default"]

# Reranker models
RERANKER_MODELS = {
    "default": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "small": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "medium": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "large": "cross-encoder/ms-marco-base-v2",
    "multilingual": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
}

DEFAULT_RERANKER_MODEL = RERANKER_MODELS["default"]

# Query expansion model
DEFAULT_QUERY_EXPANSION_MODEL = OLLAMA_MODELS["small"]

# Model parameters
MODEL_PARAMETERS = {
    "temperature": {
        "default": 0.1,
        "creative": 0.7,
        "factual": 0.0,
        "balanced": 0.3
    },
    "timeout": {
        "default": 30,
        "quick": 10,
        "normal": 30,
        "extended": 60,
        "long": 120
    },
    "max_tokens": {
        "default": None,  # Use model default
        "short": 256,
        "medium": 512,
        "long": 1024,
        "very_long": 2048
    }
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
    "solar:10.7b"
]

# Model capabilities
MODEL_CAPABILITIES = {
    "supports_tools": [
        "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo",
        "claude-3-opus", "claude-3-sonnet",
        "llama3.1:70b", "llama3.1:8b"
    ],
    "supports_vision": [
        "gpt-4-vision", "gpt-4-turbo",
        "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
        "llava", "bakllava"
    ],
    "supports_long_context": [
        "claude-3-opus", "claude-3-sonnet",
        "gpt-4-turbo", "gpt-4-32k",
        "yi:34b", "qwen2.5:32b"
    ]
}
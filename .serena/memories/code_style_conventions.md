# Code Style and Conventions

## Python Style
- **Python Version**: 3.13+
- **Type Hints**: Comprehensive type annotations using `typing` module
- **Docstrings**: Not extensively used in the codebase (minimal documentation)
- **Naming**: 
  - Classes: PascalCase (e.g., `RAGSystem`, `LLMManager`)
  - Functions/methods: snake_case (e.g., `create_database`, `handle_error`)
  - Constants: UPPER_SNAKE_CASE (e.g., `DEFAULT_MODEL`, `DEFAULT_CHUNK_SIZE`)
  - Private methods: Leading underscore (e.g., `_initialize_llm`)

## Architecture Patterns
- **Service-Oriented**: Clean separation between UI, services, and core logic
- **Manager Pattern**: Dedicated managers for each component (LLM, vectorstore, embeddings)
- **Dependency Injection**: Components receive dependencies through constructors
- **Configuration Modules**: Settings organized into logical modules (model_config, system_config, etc.)

## Error Handling
- Use `ErrorHandler.handle_error()` for structured error info
- Provide recovery suggestions in error messages
- Log errors with `RichLogger.error()`
- Return user-friendly messages

## Import Style
- Modern LangChain imports (0.3+ pattern)
- Grouped imports: stdlib, third-party, local
- Avoid wildcard imports (except in __init__.py for re-exports)

## File Organization
- `core/`: Business logic and managers
- `services/`: Service layer coordinating core components
- `config/`: Configuration constants
- `ui/`: User interface components
- `utils/`: Shared utilities
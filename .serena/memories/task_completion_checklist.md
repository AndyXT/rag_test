# Task Completion Checklist

When completing a coding task in this project, ensure:

## 1. Code Quality Checks
```bash
# Run ruff linter - MUST PASS without errors
ruff check

# Format code if needed
ruff format
```

## 2. Test Changes
```bash
# Run the application to verify it works
python -m rag_cli.main
```

## 3. Architecture Compliance
- Ensure changes follow service-oriented architecture
- Maintain separation between UI, services, and core logic
- Use appropriate manager classes for component-specific logic
- Follow dependency injection patterns

## 4. Error Handling
- All errors handled with ErrorHandler
- User-friendly error messages provided
- Recovery suggestions included where applicable
- Proper logging with RichLogger

## 5. Import Patterns
- Use modern LangChain imports (langchain_* packages)
- No unnecessary wildcard imports
- Proper re-exports in __init__.py files

## 6. Documentation
- Update CLAUDE.md if architecture changes
- Ensure code comments for complex logic
- Update type hints for new methods

## Important: Always run `ruff check` before considering task complete!
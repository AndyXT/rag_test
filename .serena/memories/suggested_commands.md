# Suggested Commands for RAG Chat Development

## Running the Application
```bash
# Main entry point (recommended)
python -m rag_cli.main

# Alternative
python rag_cli/main.py

# Legacy entry
python rag_cli.py
```

## Development Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Install with pyproject.toml (editable)
pip install -e .

# Using uv package manager
uv pip install -r requirements.txt
```

## Code Quality
```bash
# Run ruff linter
ruff check

# Fix auto-fixable issues
ruff check --fix

# Format code with ruff
ruff format
```

## System Commands (Darwin/macOS)
```bash
# List files
ls -la

# Find files
find . -name "*.py"

# Search in files
grep -r "pattern" .

# Git commands
git status
git diff
git add .
git commit -m "message"
```

## Ollama Setup (for local LLM)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull default model
ollama pull llama3.2:3b

# Start service
ollama serve
```
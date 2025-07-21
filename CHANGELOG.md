# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Import modernisation

### Changed
- **BREAKING**: Updated LangChain dependencies to 0.3+ compatibility versions
- Modernized import statements following LangChain 0.2+ patterns:
  - `langchain_community.embeddings.HuggingFaceEmbeddings` → `langchain_huggingface.HuggingFaceEmbeddings`
  - `langchain_community.vectorstores.Chroma` → `langchain_chroma.Chroma`
  - `langchain_community.llms.Ollama` → `langchain_ollama.OllamaLLM`
- Replaced deprecated `RetrievalQA.from_chain_type()` with modern `create_retrieval_chain()` approach
- Updated prompt handling from `PromptTemplate` to `ChatPromptTemplate`
- Changed chain execution from `.run()` to `.invoke()` method

### Added
- New package dependencies:
  - `langchain-huggingface` - For HuggingFace embeddings
  - `langchain-chroma` - For Chroma vector store integration  
  - `langchain-ollama` - For Ollama LLM integration

### Technical Notes
- **Environment Variables**: No new environment variables required
- **Migration**: Existing ChromaDB databases remain compatible
- **Dependencies**: Updated minimum versions:
  - `langchain>=0.3.0`
  - `langchain-community>=0.3.0`
  - `langchain-core>=0.3.0`
  - `langchain-text-splitters>=0.3.0`
  - `langchain-openai>=0.2.0`

## [0.1.0] - Initial release

### Added
- RAG Chat application with Textual TUI interface
- Support for PDF document ingestion
- ChromaDB vector storage
- Ollama LLM integration
- Basic chat interface with document querying

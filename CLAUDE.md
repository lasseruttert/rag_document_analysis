# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **RAG (Retrieval-Augmented Generation) Document Analysis System** built in Python. The system processes text documents, creates semantic embeddings, stores them in a vector database, and provides intelligent question-answering capabilities through a web interface.

## Development Commands

### Environment Setup
```bash
# Install dependencies
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pandas numpy scikit-learn jupyter
pip install sentence-transformers chromadb langchain pypdf2 python-docx python-dotenv streamlit
pip install pdfplumber>=3.0.0 PyMuPDF>=1.23.0
```

### Testing and Validation
```bash
# Test all dependencies and system setup
python test.py

# Test individual components
python src/text_processor.py      # Text processing and chunking
python src/embeddings.py          # Embedding generation  
python src/vectorstore.py         # Vector database operations
python src/retriever.py           # Semantic retrieval
python src/llm_handler.py         # LLM response generation
python src/pipeline.py            # End-to-end RAG pipeline
```

### Running the Application
```bash
# Start Streamlit web interface
streamlit run app/streamlit_app.py
```

## Architecture Overview

### Core Pipeline Flow
1. **Document Ingestion**: Multi-format files (TXT/PDF/DOCX) → Text extraction → Chunking (1000 chars, 200 overlap) → Embeddings (384D) → ChromaDB storage
2. **Query Processing**: User query → Query embedding → Semantic search → Context retrieval → T5 generation → Response

### Key Components

**Text Processing (`src/text_processor.py`)**:
- **Multi-format support**: TXT, PDF (via pdfplumber + PyMuPDF fallback), DOCX (via python-docx)
- **Advanced PDF parsing**: Table extraction, multi-page support, fallback strategies
- **DOCX processing**: Header preservation, table extraction, element-order retention
- Character-based chunking with configurable overlap
- Enhanced metadata tracking (filename, file_type, position, chunk_id, chunk_size)
- File size validation (50MB limit) and error handling

**Embedding Management (`src/embeddings.py`)**:
- Uses `sentence-transformers/all-MiniLM-L6-v2` model
- Automatic GPU/CPU detection
- Batch processing for performance (default batch_size=32)

**Vector Store (`src/vectorstore.py`)**:
- ChromaDB persistent storage in `data/vectordb/`
- Collection-based document organization
- Automatic directory creation and management

**Retriever (`src/retriever.py`)**:
- Semantic similarity search with distance scoring
- Top-K retrieval (default k=5)
- Formatted results with metadata preservation

**LLM Handler (`src/llm_handler.py`)**:
- T5-based generation (`google/flan-t5-base`)
- German language prompts and responses
- Context-aware answer generation with fallbacks
- Beam search optimization (num_beams=5)

**RAG Pipeline (`src/pipeline.py`)**:
- Orchestrates all components
- Handles document ingestion and query processing
- Configurable model and processing parameters

**Streamlit App (`app/streamlit_app.py`)**:
- **Enhanced file upload**: Support for TXT, PDF, DOCX with type validation
- **Improved UI**: File type icons, upload statistics, progress bars
- **Advanced context display**: Expandable chunks with file type indicators
- Real-time document ingestion with progress tracking
- Enhanced metadata display (file type distribution, chunk statistics)
- Cached pipeline for performance

## Data Management

### Document Storage
- **Input**: `data/raw_texts/` - Place TXT, PDF, or DOCX files here for processing
- **Processed**: `data/processed/` - Currently unused, reserved for processed data
- **Vector DB**: `data/vectordb/` - ChromaDB persistent storage with UUID-based indices
- **File type support**: .txt, .pdf, .docx with automatic format detection

### Current Test Documents
- `kubernetes_basics.txt`: Kubernetes fundamentals (Pods, Services, Deployments)
- `python_best_practices.txt`: Python coding best practices (PEP 8, virtual environments, testing)

## Configuration and Customization

### Model Configuration
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **LLM Model**: `google/flan-t5-base`
- **Chunking**: 1000 characters with 200 character overlap
- **Retrieval**: Top-5 most similar chunks

### Performance Settings
- **Token Limits**: Input 2048, Output 512 tokens
- **GPU Support**: Automatic CUDA detection and utilization
- **Batch Processing**: Configurable batch sizes for embeddings

## Development Patterns

### Module Independence
Each component in `src/` is designed to be independently testable and executable. Run any module directly to see its functionality with test data.

### Error Handling
- Components include fallback mechanisms for missing context
- German language error messages in user-facing components
- Graceful degradation when GPU is unavailable

### Testing Strategy  
- Use `test.py` to verify all dependencies, GPU availability, and document parsers
- **Enhanced testing**: Multi-format document processing, metadata validation, file size limits
- Each module includes `__main__` blocks with sample usage and format-specific testing
- **System integration tests**: End-to-end validation with new file types
- Test with provided sample documents before adding new content

## Integration Points

### Adding New Document Types
✅ **COMPLETED**: PDF/DOCX support implemented with dual parsing strategies:
- **PDF**: pdfplumber (primary) + PyMuPDF (fallback) with table extraction
- **DOCX**: python-docx with structure preservation and table support
- **TXT**: Enhanced with encoding fallbacks and validation

To add more formats, extend the `load_all_files()` function in `src/text_processor.py`.

### Improving Retrieval
Consider hybrid approaches combining semantic and keyword search (PLAN.md Phase 2.1).

### Scaling Considerations
Current system works well for documents up to several MB. For larger corpora, implement batch processing and caching strategies from PLAN.md Phase 6.

## Future Development Reference

See `PLAN.md` for comprehensive enhancement roadmap including:
- Advanced file format support (PDF/DOCX)
- Hybrid retrieval systems
- Multi-model LLM integration
- Evaluation frameworks
- Performance optimizations

The system is production-ready for local document analysis with **comprehensive multi-format support** (TXT, PDF, DOCX) and can be extended systematically following the planned phases.

## Coding Best Practices

### Code Quality Standards

**Type Hints and Documentation**:
- Always use type hints for function parameters and return values
- Use `from typing import List, Dict, Optional, Union, Any` imports
- Add comprehensive docstrings using Google/NumPy style
- Document complex algorithms and non-obvious business logic

```python
from typing import List, Dict, Optional
def process_documents(directory_path: str, chunk_size: int = 1000) -> List[Dict[str, Any]]:
    """
    Process all text documents from a directory into chunks.
    
    Args:
        directory_path: Path to directory containing text files
        chunk_size: Maximum characters per chunk
        
    Returns:
        List of document chunks with metadata
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If chunk_size is invalid
    """
```

**Error Handling and Validation**:
- Use specific exception types, never bare `except:`
- Validate inputs at function entry points
- Provide meaningful error messages in German for user-facing components
- Log errors appropriately with context information
- Use early returns to reduce nesting

```python
def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not chunks:
        raise ValueError("Chunks list cannot be empty")
    
    if not isinstance(chunks, list):
        raise TypeError(f"Expected list, got {type(chunks)}")
    
    try:
        embeddings = self.model.encode(contents)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise RuntimeError(f"Failed to generate embeddings: {e}") from e
```

**Resource Management**:
- Use context managers (`with` statements) for file operations and database connections
- Implement proper cleanup in classes with `__enter__` and `__exit__` methods
- Handle GPU memory management explicitly
- Close database connections and file handles properly

```python
class VectorStoreManager:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()
```

### Performance and Scalability

**Memory Management**:
- Process large files in chunks rather than loading entirely into memory
- Use generators for large data processing pipelines
- Clear unnecessary variables and call `gc.collect()` for memory-intensive operations
- Monitor GPU memory usage and implement batch size adjustments

```python
def process_large_file(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """Process large files in chunks to avoid memory issues."""
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            yield process_chunk(chunk)
```

**Batch Processing**:
- Always use batch processing for ML model operations
- Implement configurable batch sizes with sensible defaults
- Add progress bars for long-running operations
- Handle partial batch failures gracefully

**Caching Strategy**:
- Cache expensive computations (embeddings, model outputs)
- Use appropriate cache invalidation strategies
- Consider disk-based caching for large datasets
- Implement cache size limits and LRU eviction

### Code Organization and Modularity

**Separation of Concerns**:
- Keep data processing, business logic, and presentation layers separate
- Each class should have a single, well-defined responsibility
- Use dependency injection for better testability
- Avoid circular imports between modules

**Configuration Management**:
- Centralize configuration in `src/config.py`
- Use environment variables for deployment-specific settings
- Validate configuration at startup
- Provide sensible defaults for all configuration options

```python
@dataclass
class RAGConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "google/flan-t5-base"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    batch_size: int = 32
    use_gpu: bool = True
    
    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
```

**Interface Design**:
- Define clear, consistent interfaces between components
- Use abstract base classes for components that might have multiple implementations
- Keep public APIs stable and document breaking changes
- Prefer composition over inheritance

### Testing and Reliability

**Comprehensive Testing**:
- Write unit tests for all core functionality
- Use pytest with proper fixtures and parametrization
- Mock external dependencies (models, databases, APIs)
- Test error conditions and edge cases
- Maintain test coverage above 80%

```python
@pytest.fixture
def sample_documents():
    return [
        {"content": "Test content", "metadata": {"filename": "test.txt"}},
        {"content": "Another test", "metadata": {"filename": "test2.txt"}}
    ]

def test_embedding_generation(sample_documents):
    embedder = EmbeddingManager(model_name="test-model")
    with pytest.patch.object(embedder.model, 'encode') as mock_encode:
        mock_encode.return_value = np.array([[0.1, 0.2, 0.3]])
        result = embedder.generate_embeddings(sample_documents)
        assert len(result) == len(sample_documents)
        assert 'embedding' in result[0]
```

**Logging and Monitoring**:
- Use structured logging with appropriate log levels
- Log performance metrics (processing times, batch sizes)
- Include correlation IDs for request tracking
- Log errors with full context and stack traces

```python
import logging
logger = logging.getLogger(__name__)

def process_query(self, query: str) -> str:
    start_time = time.time()
    logger.info(f"Processing query: {query[:50]}...")
    
    try:
        result = self._generate_response(query)
        duration = time.time() - start_time
        logger.info(f"Query processed successfully in {duration:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise
```

### Security and Data Protection

**Input Validation**:
- Sanitize all user inputs, especially file uploads
- Validate file types and sizes before processing
- Prevent path traversal attacks in file operations
- Use allowlists rather than blocklists for validation

**Data Privacy**:
- Never log sensitive user data or document contents
- Implement secure deletion of temporary files
- Consider data retention policies for vector databases
- Ensure GDPR compliance for document processing

**Dependency Management**:
- Pin dependency versions to avoid supply chain attacks
- Regularly update dependencies and scan for vulnerabilities
- Use virtual environments to isolate dependencies
- Document all external dependencies and their purposes

### Performance Optimization

**Profiling and Benchmarking**:
- Use `cProfile` and `line_profiler` to identify bottlenecks
- Benchmark critical operations with realistic data
- Monitor memory usage with `memory_profiler`
- Set performance baselines and regression tests

**Database Optimization**:
- Use appropriate indexing strategies in ChromaDB
- Implement connection pooling for concurrent access
- Monitor query performance and optimize slow operations
- Consider data partitioning for large document collections

**Model Optimization**:
- Use model quantization when appropriate
- Implement model warming to avoid cold start penalties
- Consider model distillation for production deployments
- Use ONNX or TensorRT for inference optimization

### Maintainability

**Code Review Standards**:
- All code changes require review before merging
- Use consistent naming conventions (snake_case for functions/variables)
- Keep functions small and focused (ideally <50 lines)
- Remove dead code and unused imports regularly

**Documentation**:
- Maintain up-to-date README with setup instructions
- Document architectural decisions in design docs
- Keep inline comments focused on "why" not "what"
- Update documentation when changing APIs

**Refactoring Guidelines**:
- Refactor proactively when adding new features
- Extract common patterns into reusable utilities
- Remove code duplication through proper abstraction
- Keep refactoring changes separate from feature changes

This RAG system should be extended following these practices to ensure long-term maintainability, reliability, and performance.

## Development Environment Guidelines

### Environment Restrictions
- **Key Rule**: Do not install anything outside of the conda environment "rag"
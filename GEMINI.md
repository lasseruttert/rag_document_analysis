# GEMINI.md

This file provides guidance to the Gemini AI assistant when working with this repository.

## Project Overview

This is a **RAG (Retrieval-Augmented Generation) Document Analysis System** built in Python. The system processes text documents, creates semantic embeddings, stores them in a vector database (ChromaDB), and provides a question-answering interface using a Large Language Model (LLM). The user interface is built with Streamlit.

The project is well-structured, with separate modules for each part of the RAG pipeline.

## Key Technologies

- **Language:** Python
- **Core Libraries:**
    - `sentence-transformers`: For generating text embeddings.
    - `chromadb`: For the vector database.
    - `transformers`: For the LLM (T5-based).
    - `streamlit`: For the web UI.
- **Models:**
    - **Embedding Model:** `all-MiniLM-L6-v2` (384 dimensions)
    - **LLM:** `google/flan-t5-base`

## Project Structure

```
├── app/
│   └── streamlit_app.py        # Streamlit Web-Interface
├── data/
│   ├── raw_texts/              # Input text files (.txt)
│   └── vectordb/               # ChromaDB persistent storage
├── src/
│   ├── text_processor.py       # Text loading and chunking
│   ├── embeddings.py           # Embedding generation
│   ├── vectorstore.py          # ChromaDB management
│   ├── retriever.py            # Semantic search and retrieval
│   ├── llm_handler.py          # LLM interaction
│   └── pipeline.py             # Main RAG pipeline orchestration
├── test.py                     # Dependency and setup tests
├── README.md                   # Installation instructions
└── GEMINI.md                   # Guidance for the Gemini assistant
```

## Development Workflow

### 1. Environment Setup

To set up the development environment, use the commands from the `README.md`:

```bash
# Install conda packages
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pandas numpy scikit-learn jupyter

# Install pip packages
pip install sentence-transformers chromadb langchain pypdf2 python-docx python-dotenv streamlit
```

### 2. Running Tests

A `test.py` script is provided to verify the setup and dependencies.

```bash
python test.py
```

Each module in `src/` can also be run independently for testing purposes, as they contain `if __name__ == '__main__':` blocks.

### 3. Running the Application

The main application is the Streamlit web interface.

```bash
streamlit run app/streamlit_app.py
```

### 4. Document Ingestion Workflow

1.  **Add Documents:** Place new `.txt` files into the `data/raw_texts/` directory.
2.  **Run Ingestion:** Use the "Dokumente ingestieren" button in the Streamlit UI. This triggers the `RAGPipeline.ingest_documents()` method, which processes the text files, generates embeddings, and stores them in the ChromaDB vector store located at `data/vectordb/`.

### 5. Answering Questions

-   Once documents are ingested, type a question into the text input in the Streamlit UI.
-   The `RAGPipeline.answer_query()` method is called, which retrieves relevant text chunks from ChromaDB and uses the LLM to generate an answer.

## Architectural Principles

-   **Modularity:** The `src` directory contains distinct, single-responsibility modules for each stage of the RAG pipeline (processing, embedding, storing, retrieving, generating).
-   **Configuration:** The core pipeline components are initialized in `src/pipeline.py`, with model names and paths passed as arguments. Future work may involve moving this to a `config.yaml` file as per `PLAN.md`.
-   **Caching:** The Streamlit app uses `@st.cache_resource` to cache the `RAGPipeline` instance for better performance.

## Coding Best Practices

To ensure the code remains clean, readable, and maintainable, please adhere to the following guidelines when making changes:

-   **Follow PEP 8:** All Python code should conform to the PEP 8 style guide. Use a linter like `flake8` or `black` to format your code automatically.
-   **Type Hinting:** Use type hints for all function signatures (arguments and return values) as demonstrated in the existing code (`from typing import List, Dict`). This improves code clarity and allows for static analysis.
-   **Docstrings:** Write clear and concise docstrings for all modules, classes, and functions, following the Google Python Style Guide. Explain what the code does, its arguments, and what it returns.
-   **Modularity and Single Responsibility:** Continue to follow the existing modular design. Each function and class should have a single, well-defined purpose.
-   **Error Handling:** Implement robust error handling. Use `try...except` blocks to catch potential exceptions (e.g., file not found, API errors) and provide informative error messages.
-   **Configuration Management:** Avoid hardcoding values. Use the `src/config.py` module or a `config.yaml` file (as suggested in `PLAN.md`) for managing settings like model names, paths, and hyperparameters.
-   **Testing:** Any new feature or bug fix should be accompanied by tests. Add unit tests to the `if __name__ == '__main__':` blocks for simple validation or create dedicated test files in a `tests/` directory for more complex scenarios.
-   **Readability:** Write code that is easy to understand. Use meaningful variable names, keep functions short, and add comments only when the logic is complex and not self-evident.

## How to Help

-   **Code Modifications:** When asked to modify the code, be mindful of the modular structure. Changes to one component (e.g., `text_processor.py`) should be compatible with the other components it interacts with.
-   **Following the Plan:** The `PLAN.md` file outlines a detailed roadmap for future development. When adding new features, refer to this plan to ensure consistency with the project goals. For example, if asked to add PDF support, the implementation should follow the steps outlined in `P1.1` of `PLAN.md`.
-   **Dependency Management:** If new libraries are needed, they should be added to the installation instructions.
-   **Testing:** When adding or modifying functionality, consider how it can be tested, either by extending `test.py` or by adding logic to the `if __name__ == '__main__':` block of the relevant module.

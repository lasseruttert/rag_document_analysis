# RAG Document Analysis Project

## Projektübersicht

Dieses Projekt implementiert ein **Retrieval-Augmented Generation (RAG) System** zur Dokumentenanalyse. Das System ermöglicht es, Textdokumente zu ingestieren, sie semantisch durchsuchbar zu machen und auf Basis der Inhalte Fragen zu beantworten.

### Technologie-Stack
- **Python 3.x** mit PyTorch/CUDA-Unterstützung
- **Sentence Transformers** für Embedding-Generierung (Modell: all-MiniLM-L6-v2)
- **ChromaDB** als Vektordatenbank für semantische Suche
- **Hugging Face Transformers** mit T5-Modell (google/flan-t5-base) für Antwortgenerierung
- **Streamlit** für die Web-Interface

## Projektstruktur

```
rag_document_analysis/
├── README.md                    # Conda/pip Installationsanweisungen
├── test.py                      # Dependency-Test-Skript
├── app/
│   └── streamlit_app.py        # Streamlit Web-Interface
├── data/
│   ├── raw_texts/              # Eingabetextdateien
│   │   ├── kubernetes_basics.txt
│   │   └── python_best_practices.txt
│   ├── processed/              # (Leer - für verarbeitete Daten)
│   └── vectordb/              # ChromaDB Persistenz-Layer
│       ├── chroma.sqlite3
│       └── [UUID-Ordner]/     # Vektor-Indizes
└── src/                       # Hauptimplementierung
    ├── __init__.py
    ├── config.py              # (Leer - für Konfiguration)
    ├── text_processor.py      # Textverarbeitung und Chunking
    ├── embeddings.py          # Embedding-Generierung
    ├── vectorstore.py         # ChromaDB-Integration
    ├── retriever.py           # Semantische Suche
    ├── llm_handler.py         # T5-basierte Antwortgenerierung
    └── pipeline.py            # RAG-Pipeline-Orchestrierung
```

## Implementierte Module

### 1. Text Processing (`src/text_processor.py`)

**Funktionalität:**
- Lädt .txt-Dateien aus einem Verzeichnis
- Chunking von Texten in überlappende Segmente (Standard: 1000 Zeichen mit 200 Zeichen Überlappung)
- Metadata-Management für jeden Chunk (Dateiname, Position, Chunk-ID)

**Wichtige Funktionen:**
- `load_text_files(directory_path)`: Lädt alle .txt-Dateien
- `chunk_text(text, filename, chunk_size, chunk_overlap)`: Teilt Text in Chunks
- `process_documents(directory_path)`: Vollständige Dokumentenverarbeitung

**Status:** ✅ Vollständig implementiert mit Hauptfunktion für Tests

### 2. Embedding Management (`src/embeddings.py`)

**Funktionalität:**
- Initialisiert Sentence Transformer Modell (all-MiniLM-L6-v2)
- GPU/CPU-automatische Erkennung und Konfiguration
- Batch-weise Embedding-Generierung für Performance-Optimierung
- Embedding-Dimension: 384

**Wichtige Funktionen:**
- `EmbeddingManager.__init__(model_name, use_gpu)`: Modell-Initialisierung
- `generate_embeddings(chunks, batch_size)`: Batch-Embedding-Generierung

**Status:** ✅ Vollständig implementiert mit Test-Pipeline

### 3. Vector Store Management (`src/vectorstore.py`)

**Funktionalität:**
- ChromaDB-Integration mit persistenter Speicherung
- Collection-Management für verschiedene Dokumentensets
- Automatic Directory-Erstellung für Datenbankpfad
- Embedding-Storage mit Metadaten

**Wichtige Funktionen:**
- `VectorStoreManager.__init__(storage_path)`: ChromaDB-Client-Initialisierung
- `create_or_get_collection(name)`: Collection-Management
- `add_documents(chunks_with_embeddings)`: Dokument-Speicherung

**Status:** ✅ Vollständig implementiert mit Verifikations-Logik

### 4. Semantic Retrieval (`src/retriever.py`)

**Funktionalität:**
- Query-Embedding-Generierung
- Semantische Ähnlichkeitssuche in ChromaDB
- Top-K Retrieval mit Distanz-Scoring
- Formatierte Ergebnis-Rückgabe mit Metadaten

**Wichtige Funktionen:**
- `retrieve(query, top_k)`: Hauptretrieval-Funktion
- Automatische Query-zu-Vektor-Konvertierung
- Distanz-basierte Relevanz-Bewertung

**Status:** ✅ Vollständig implementiert mit Test-Queries

### 5. LLM Handler (`src/llm_handler.py`)

**Funktionalität:**
- T5-Modell-Integration (google/flan-t5-base)
- Context-Aware Antwortgenerierung
- Deutschsprachige Prompt-Templates
- Beam Search für verbesserte Antwortqualität

**Wichtige Funktionen:**
- `generate_answer(query, context_chunks)`: Hauptantwort-Generierung
- Context-Aggregation aus multiplen Chunks
- Fallback für fehlenden Kontext

**Konfiguration:**
- Max Input Length: 2048 Tokens
- Max Output Length: 512 Tokens
- Beam Search: 5 Beams
- Early Stopping aktiviert

**Status:** ✅ Vollständig implementiert mit End-to-End Test

### 6. RAG Pipeline (`src/pipeline.py`)

**Funktionalität:**
- Orchestrierung aller RAG-Komponenten
- Einmalige Dokumenten-Ingestion
- End-to-End Query-Processing
- Konfigurierbare Modell-Parameter

**Pipeline-Schritte:**
1. Dokumenten-Verarbeitung → Chunking
2. Embedding-Generierung
3. Vektordatenbank-Speicherung
4. Query-Processing → Retrieval → LLM-Generierung

**Wichtige Funktionen:**
- `ingest_documents()`: Vollständige Dokumenten-Pipeline
- `answer_query(query, top_k)`: End-to-End Antwortgenerierung

**Status:** ✅ Vollständig implementiert mit Beispiel-Queries

### 7. Streamlit Web Interface (`app/streamlit_app.py`)

**Funktionalität:**
- Web-basierte Benutzeroberfläche
- Datei-Upload für .txt-Dokumente
- Dokumenten-Ingestion-Trigger
- Echtzeit Query-Interface mit Kontext-Anzeige

**Features:**
- Sidebar für Dokumenten-Management
- Upload-Verzeichnis: `data/raw_texts`
- Cached RAG-Pipeline für Performance
- Transparenz durch Kontext-Chunk-Anzeige mit Distanz-Scores

**Status:** ✅ Vollständig implementiert mit caching

## Datenbasis

### Aktuell verfügbare Dokumente:
1. **kubernetes_basics.txt**: Kubernetes-Grundlagen (Pods, Services, Deployments, ConfigMaps)
2. **python_best_practices.txt**: Python-Best-Practices (PEP 8, virtuelle Umgebungen, Type Hints, Testing)

### Vektordatenbank:
- **Typ:** ChromaDB persistent storage
- **Speicherort:** `data/vectordb/`
- **Collections:** Separate Collections für verschiedene Dokumentensets möglich
- **Index-Status:** 4 UUID-basierte Indizes vorhanden (vermutlich für verschiedene Dokumenten-Chunks)

## Setup und Installation

### Dependencies (aus README.md):
```bash
# Conda Packages
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pandas numpy scikit-learn jupyter

# Pip Packages
pip install sentence-transformers chromadb langchain
pip install pypdf2 python-docx python-dotenv streamlit
```

### Test-Skript (`test.py`):
- Überprüft alle kritischen Dependencies
- GPU/CUDA-Verfügbarkeit-Test
- Sentence Transformer Modell-Download-Test
- Embedding-Dimensions-Verifikation (384D)

## Architektur-Design

### RAG-Flow:
1. **Ingestion Phase:**
   - Text-Dokumente → Chunking → Embeddings → Vektordatenbank

2. **Query Phase:**
   - User Query → Query Embedding → Ähnlichkeitssuche → Top-K Retrieval → LLM Context → Antwortgenerierung

### Modularität:
- Jede Komponente ist eigenständig testbar
- Klare Trennung zwischen Verarbeitung, Speicherung und Generierung
- Dependency Injection für flexible Konfiguration

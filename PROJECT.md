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

### 1. Text Processing (`src/text_processor.py`) 🆕 **ERWEITERT**

**Funktionalität:**
- **Multi-Format-Support**: TXT, PDF, DOCX mit automatischer Format-Erkennung
- **PDF-Verarbeitung**: Dual-Parser-Strategie (pdfplumber + PyMuPDF fallback)
- **DOCX-Verarbeitung**: Struktur-erhaltende Extraktion mit python-docx
- **Tabellen-Extraktion**: Strukturierte Konvertierung von PDF/DOCX-Tabellen
- Chunking von Texten in überlappende Segmente (Standard: 1000 Zeichen mit 200 Zeichen Überlappung)
- **Erweiterte Metadaten**: Dateiname, Dateityp, Position, Chunk-ID, Chunk-Größe
- **Dateigröße-Validierung**: 50MB Limit mit Logging
- **Robuste Error-Behandlung**: Encoding-Fallbacks, Permission-Handling

**Wichtige Funktionen:**
- `load_all_files(directory_path)`: Lädt alle unterstützten Dateitypen
- `load_pdf_files(directory_path)`: PDF-spezifische Verarbeitung
- `load_docx_files(directory_path)`: DOCX-spezifische Verarbeitung
- `load_text_files_extended(directory_path)`: Erweiterte TXT-Verarbeitung
- `chunk_text(text, filename, chunk_size, chunk_overlap, file_type)`: Erweiterte Chunking-Funktion
- `process_documents(directory_path)`: Vollständige Multi-Format-Dokumentenverarbeitung
- `validate_file_size(file_path)`: Dateigröße-Validierung
- `safe_file_processing(file_path, processor_func)`: Sichere Dateiverarbeitung

**Status:** ✅ Vollständig implementiert mit Multi-Format-Support und umfassendem Testing

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

### 7. Streamlit Web Interface (`app/streamlit_app.py`) 🆕 **ERWEITERT**

**Funktionalität:**
- Web-basierte Benutzeroberfläche mit verbesserter UX
- **Multi-Format-Upload**: TXT, PDF, DOCX mit Typ-Validierung
- **Enhanced Upload-Feedback**: Typ-Icons, Upload-Statistiken, Fehler-Handling
- Dokumenten-Ingestion mit Progress-Bars und Status-Updates
- **Erweiterte Query-Interface**: Expandable Context-Chunks mit Metadaten

**Features:**
- **Smart Sidebar**: Dokumenttyp-Statistiken, Upload-Zusammenfassung
- Upload-Verzeichnis: `data/raw_texts` (Multi-Format)
- **Progress-Tracking**: Echtzeit-Status für Ingestion-Prozess
- **Enhanced Context-Display**: File-Type-Icons, Relevanz-Scores, Chunk-Details
- **Metadata-Rich UI**: Position, Chunk-Größe, Distanz-Metriken
- Cached RAG-Pipeline für Performance

**Status:** ✅ Vollständig implementiert mit Multi-Format-Support und verbesserter UX

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
pip install pdfplumber>=3.0.0 PyMuPDF>=1.23.0  # 🆕 Neu hinzugefügt
```

### Test-Skript (`test.py`) 🆕 **ERWEITERT**:
- Überprüft alle kritischen Dependencies (inkl. pdfplumber, PyMuPDF)
- GPU/CUDA-Verfügbarkeit-Test
- **Dokumenten-Parser-Tests**: PDF/DOCX-Parsing-Fähigkeiten
- **Erweiterte Dokumentverarbeitung**: Multi-Format-Processing-Tests
- **Dateityp-Validierung**: Größen-Limits und Error-Handling
- **System-Integration-Test**: End-to-End-Validierung mit neuen Metadaten
- Sentence Transformer Modell-Download-Test
- Embedding-Dimensions-Verifikation (384D)

## Architektur-Design

### RAG-Flow: 🆕 **ERWEITERT**
1. **Ingestion Phase:**
   - **Multi-Format-Dokumente** (TXT/PDF/DOCX) → **Format-spezifische Extraktion** → Chunking → Embeddings → Vektordatenbank (mit Dateityp-Metadaten)

2. **Query Phase:**
   - User Query → Query Embedding → Ähnlichkeitssuche → Top-K Retrieval → **Enhanced Context** (mit Dateityp-Info) → LLM Context → Antwortgenerierung

### Modularität:
- Jede Komponente ist eigenständig testbar
- Klare Trennung zwischen Verarbeitung, Speicherung und Generierung
- Dependency Injection für flexible Konfiguration

## Aktuelle Einschränkungen 🆕 **REDUZIERT**

1. ~~**Textformat-Limitation:** Nur .txt-Dateien unterstützt~~ ✅ **BEHOBEN**: Vollständiger Multi-Format-Support (TXT/PDF/DOCX)
2. **Sprachmodell:** T5-base ist relativ klein für komplexe Reasoning-Aufgaben
3. **Chunking-Strategie:** Simple character-based chunking ohne semantische Grenzen *(Nächste Priorität: P1.2)*
4. **Evaluation:** Keine Metriken für Retrieval-Qualität oder Antwort-Evaluation
5. **Neue Einschränkungen:**
   - **Dateigröße-Limit:** 50MB pro Datei
   - **Bildinhalt:** PDFs/DOCX mit Bildern/Diagrammen werden nur als Text extrahiert
   - **Komplexe Layouts:** Sehr spezielle PDF-Formate könnten Parsing-Probleme haben

## Implementierungs-Status 🆕

### ✅ Abgeschlossen (Januar 2025):
- **P1.1 Multi-Format-Support**: PDF/DOCX/TXT-Verarbeitung vollständig implementiert
- **Enhanced UI**: File-Type-Icons, Progress-Tracking, Metadaten-Display
- **Robuste Error-Behandlung**: Größen-Validierung, Encoding-Fallbacks, Logging
- **Comprehensive Testing**: Multi-Format-Tests, System-Integration, Metadaten-Validierung

### 🔄 Nächste Priorität:
- **P1.2 Verbessertes Chunking**: Semantisches Chunking mit spaCy/NLTK (siehe PLAN.md)
- **P1.3 Konfigurationssystem**: YAML-basierte Einstellungen ohne Code-Änderungen

## Zusammenfassung

**Projektreife:** Production-ready für lokale Dokumentenanalyse mit **vollständigem Multi-Format-Support** (TXT/PDF/DOCX) und erweiterten UI-Features. Das System ist bereit für den produktiven Einsatz und systematische Erweiterung gemäß PLAN.md.

**Wichtigste Verbesserungen (Januar 2025):**
- 📄 **Multi-Format-Verarbeitung**: PDF (pdfplumber + PyMuPDF) und DOCX (python-docx) vollständig integriert
- 🗂️ **Tabellen-Extraktion**: Strukturierte Konvertierung von PDF/DOCX-Tabellen zu Text
- 🎯 **Enhanced UI**: File-Type-Icons, Upload-Statistiken, Progress-Tracking, erweiterte Metadaten
- 🛡️ **Robuste Error-Behandlung**: 50MB File-Size-Limits, Encoding-Fallbacks, umfassendes Logging
- ✅ **Comprehensive Testing**: Multi-Format-Tests, System-Integration, Metadaten-Validierung

Das RAG-System unterstützt jetzt alle gängigen Dokumentformate und bietet eine professionelle, benutzerfreundliche Oberfläche für die Dokumentenanalyse.

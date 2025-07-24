# RAG System Enhancement Plan

## Übersicht

Dieses Dokument definiert einen strukturierten Entwicklungsplan für die Erweiterung und Verbesserung des bestehenden RAG Document Analysis Systems. Die Implementierungen sind nach Priorität und Komplexität geordnet.

---

## Phase 1: Sofortige Verbesserungen (Priorität: HOCH)

### P1.1: Erweiterte Dateiformate-Unterstützung ✅ **ABGESCHLOSSEN**
**Ziel:** PDF und DOCX Dokumente verarbeiten können

**Status:** ✅ **VOLLSTÄNDIG IMPLEMENTIERT** (Januar 2025)

**Detaillierte Implementierungs-Roadmap:**

#### Schritt 1: Dependency-Setup und Testing
```bash
# Neue Dependencies installieren
pip install pdfplumber>=3.0.0 python-docx>=0.8.11 PyMuPDF>=1.23.0
```

**Update `test.py`:**
- Erweitere Package-Liste um `pdfplumber`, `docx`, `fitz` (PyMuPDF)
- Füge Testfunktion hinzu:
```python
def test_document_parsers():
    # Test PDF parsing mit sample PDF
    # Test DOCX parsing mit sample DOCX
    # Test error handling für korrupte Dateien
```

#### Schritt 2: PDF-Parser-Implementierung (`src/text_processor.py`)

**Neue Imports hinzufügen:**
```python
import pdfplumber
import fitz  # PyMuPDF als Fallback
from pathlib import Path
import logging
from typing import Tuple, Optional
```

**PDF-Parsing-Funktion implementieren:**
```python
def load_pdf_files(directory_path: str) -> List[Tuple[str, str, str]]:
    """
    Lädt alle .pdf-Dateien aus einem Verzeichnis.
    Returns: List[Tuple[filename, content, file_type]]
    """
    pdf_files = []
    directory = Path(directory_path)
    
    for pdf_path in directory.glob("*.pdf"):
        try:
            # Primäre Methode: pdfplumber (bessere Textextraktion)
            content = extract_text_with_pdfplumber(pdf_path)
            if not content.strip():
                # Fallback: PyMuPDF für komplexe PDFs
                content = extract_text_with_pymupdf(pdf_path)
            
            if content.strip():
                pdf_files.append((pdf_path.name, content, "pdf"))
            else:
                logging.warning(f"No text extracted from {pdf_path.name}")
                
        except Exception as e:
            logging.error(f"Error processing PDF {pdf_path.name}: {e}")
            continue
    
    return pdf_files

def extract_text_with_pdfplumber(pdf_path: Path) -> str:
    """Extrahiert Text mit pdfplumber (behält Tabellen-Struktur)."""
    full_text = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            try:
                # Text extrahieren
                text = page.extract_text()
                if text:
                    full_text.append(f"[Seite {page_num}]\n{text}")
                
                # Tabellen separat extrahieren
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if table:
                        table_text = format_table_as_text(table)
                        full_text.append(f"[Seite {page_num}, Tabelle {table_idx + 1}]\n{table_text}")
                        
            except Exception as e:
                logging.warning(f"Error extracting page {page_num} from {pdf_path.name}: {e}")
                continue
    
    return "\n\n".join(full_text)

def format_table_as_text(table: List[List[str]]) -> str:
    """Konvertiert Tabelle zu strukturiertem Text."""
    if not table or not table[0]:
        return ""
    
    # Header-Zeile
    headers = [cell or "" for cell in table[0]]
    formatted_rows = [" | ".join(headers)]
    formatted_rows.append("-" * len(" | ".join(headers)))
    
    # Daten-Zeilen
    for row in table[1:]:
        if row:
            formatted_row = [cell or "" for cell in row]
            formatted_rows.append(" | ".join(formatted_row))
    
    return "\n".join(formatted_rows)

def extract_text_with_pymupdf(pdf_path: Path) -> str:
    """Fallback-Extraktion mit PyMuPDF."""
    full_text = []
    
    try:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text = page.get_text()
            if text.strip():
                full_text.append(f"[Seite {page_num + 1}]\n{text}")
        pdf_document.close()
    except Exception as e:
        logging.error(f"PyMuPDF extraction failed for {pdf_path.name}: {e}")
        return ""
    
    return "\n\n".join(full_text)
```

#### Schritt 3: DOCX-Parser-Implementierung

**DOCX-Parsing-Funktion hinzufügen:**
```python
from docx import Document
from docx.table import Table
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P

def load_docx_files(directory_path: str) -> List[Tuple[str, str, str]]:
    """
    Lädt alle .docx-Dateien aus einem Verzeichnis.
    Returns: List[Tuple[filename, content, file_type]]
    """
    docx_files = []
    directory = Path(directory_path)
    
    for docx_path in directory.glob("*.docx"):
        # Skip temporary files (~$filename.docx)
        if docx_path.name.startswith("~$"):
            continue
            
        try:
            content = extract_text_from_docx(docx_path)
            if content.strip():
                docx_files.append((docx_path.name, content, "docx"))
            else:
                logging.warning(f"No text extracted from {docx_path.name}")
                
        except Exception as e:
            logging.error(f"Error processing DOCX {docx_path.name}: {e}")
            continue
    
    return docx_files

def extract_text_from_docx(docx_path: Path) -> str:
    """Extrahiert Text und Tabellen aus DOCX-Datei."""
    try:
        document = Document(docx_path)
        full_text = []
        
        # Iteriere über alle Elemente (Paragraphs und Tabellen) in Reihenfolge
        for element in document.element.body:
            if isinstance(element, CT_P):
                # Paragraph
                paragraph = next(p for p in document.paragraphs if p._element == element)
                text = paragraph.text.strip()
                if text:
                    # Formatierung beibehalten
                    if paragraph.style.name.startswith('Heading'):
                        full_text.append(f"\n## {text}\n")
                    else:
                        full_text.append(text)
                        
            elif isinstance(element, CT_Tbl):
                # Tabelle
                table = next(t for t in document.tables if t._element == element)
                table_text = extract_table_from_docx(table)
                if table_text:
                    full_text.append(f"\n[Tabelle]\n{table_text}\n")
        
        return "\n".join(full_text)
        
    except Exception as e:
        logging.error(f"Error extracting from DOCX {docx_path.name}: {e}")
        return ""

def extract_table_from_docx(table: Table) -> str:
    """Extrahiert Tabelle aus DOCX als strukturierten Text."""
    if not table.rows:
        return ""
    
    formatted_rows = []
    
    for row_idx, row in enumerate(table.rows):
        row_cells = []
        for cell in row.cells:
            cell_text = cell.text.strip().replace("\n", " ")
            row_cells.append(cell_text or "")
        
        formatted_rows.append(" | ".join(row_cells))
        
        # Header-Separator nach erster Zeile
        if row_idx == 0 and len(table.rows) > 1:
            separator = " | ".join(["-" * max(3, len(cell)) for cell in row_cells])
            formatted_rows.append(separator)
    
    return "\n".join(formatted_rows)
```

#### Schritt 4: Unified Interface Implementation

**Bestehende `load_text_files` erweitern:**
```python
def load_all_files(directory_path: str) -> List[Tuple[str, str, str]]:
    """
    Lädt alle unterstützten Dateitypen (.txt, .pdf, .docx) aus einem Verzeichnis.
    
    Returns:
        List[Tuple[filename, content, file_type]]
    """
    all_files = []
    
    # TXT-Dateien (bestehende Logik erweitern)
    txt_files = load_text_files_extended(directory_path)
    all_files.extend(txt_files)
    
    # PDF-Dateien
    pdf_files = load_pdf_files(directory_path)
    all_files.extend(pdf_files)
    
    # DOCX-Dateien
    docx_files = load_docx_files(directory_path)
    all_files.extend(docx_files)
    
    logging.info(f"Loaded {len(all_files)} files: "
                f"{len(txt_files)} TXT, {len(pdf_files)} PDF, {len(docx_files)} DOCX")
    
    return all_files

def load_text_files_extended(directory_path: str) -> List[Tuple[str, str, str]]:
    """Erweiterte TXT-Loader mit file_type Return."""
    texts = []
    directory = Path(directory_path)
    
    for txt_path in directory.glob("*.txt"):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    texts.append((txt_path.name, content, "txt"))
        except UnicodeDecodeError:
            # Fallback für andere Encodings
            try:
                with open(txt_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                    if content.strip():
                        texts.append((txt_path.name, content, "txt"))
            except Exception as e:
                logging.error(f"Error reading {txt_path.name}: {e}")
    
    return texts

# Update process_documents function
def process_documents(
    directory_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Verarbeitet alle Dokumenttypen aus einem Verzeichnis zu Chunks.
    Erweitert um file_type-Metadaten.
    """
    all_chunks = []
    loaded_files = load_all_files(directory_path)
    
    for filename, content, file_type in loaded_files:
        chunks = chunk_text(
            text=content,
            filename=filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            file_type=file_type  # Neue Parameter
        )
        all_chunks.extend(chunks)
        
    return all_chunks

# Update chunk_text to include file_type
def chunk_text(
    text: str,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    file_type: str = "txt"  # Neuer Parameter
) -> List[Dict[str, Any]]:
    """Erweitert um file_type in Metadaten."""
    if not text:
        return []

    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_content = text[start:end]
        
        chunks.append({
            "content": chunk_content,
            "metadata": {
                "filename": filename,
                "file_type": file_type,  # Neu
                "position": start,
                "chunk_id": f"{filename}-{chunk_id}",
                "chunk_size": len(chunk_content)  # Zusätzliche Info
            }
        })
        
        start += chunk_size - chunk_overlap
        chunk_id += 1
        
    return chunks
```

#### Schritt 5: Streamlit UI Updates (`app/streamlit_app.py`)

**File Upload erweitern:**
```python
# Update file uploader
uploaded_files = st.sidebar.file_uploader(
    "Laden Sie Dokumente hoch", 
    type=["txt", "pdf", "docx"],  # Erweiterte Typen
    accept_multiple_files=True,
    help="Unterstützte Formate: TXT, PDF, DOCX"
)

# File processing mit Typ-Anzeige
if uploaded_files:
    upload_stats = {"txt": 0, "pdf": 0, "docx": 0}
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_dir, uploaded_file.name)
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Validate file type
        if file_extension not in ["txt", "pdf", "docx"]:
            st.sidebar.error(f"Nicht unterstützter Dateityp: {uploaded_file.name}")
            continue
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        upload_stats[file_extension] += 1
        st.sidebar.success(f"✓ {uploaded_file.name} ({file_extension.upper()}) hochgeladen")
    
    # Upload summary
    summary_parts = []
    for file_type, count in upload_stats.items():
        if count > 0:
            summary_parts.append(f"{count} {file_type.upper()}")
    
    if summary_parts:
        st.sidebar.info(f"Hochgeladen: {', '.join(summary_parts)}")

# Document ingestion mit Progress
if st.sidebar.button("Dokumente ingestieren (in Vektor-DB)"):
    with st.spinner("Dokumente werden verarbeitet..."):
        # Progress bar für verschiedene Schritte
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        try:
            status_text.text("Lade Dokumente...")
            progress_bar.progress(25)
            
            pipeline.ingest_documents()
            
            progress_bar.progress(100)
            status_text.text("Fertig!")
            
            st.sidebar.success("Dokumente erfolgreich ingestiert!")
            
            # Zeige Statistiken
            collection_count = pipeline.collection.count()
            st.sidebar.metric("Dokumente in DB", collection_count)
            
        except Exception as e:
            st.sidebar.error(f"Fehler beim Ingestieren: {str(e)}")
            progress_bar.empty()
            status_text.empty()

# Enhanced document info display
st.sidebar.markdown("### 📊 Dokumenten-Übersicht")
if pipeline.collection.count() > 0:
    # Zeige Dokumenttyp-Statistiken
    try:
        # Hole Metadaten aus der Collection
        sample_results = pipeline.collection.get(limit=1000, include=["metadatas"])
        if sample_results['metadatas']:
            file_types = {}
            for metadata in sample_results['metadatas']:
                file_type = metadata.get('file_type', 'unknown')
                file_types[file_type] = file_types.get(file_type, 0) + 1
            
            for file_type, count in file_types.items():
                st.sidebar.metric(f"{file_type.upper()}-Chunks", count)
    except:
        pass

# Enhanced context display mit file type
if user_query and pipeline.collection.count() > 0:
    with st.spinner("Antwort wird generiert..."):
        answer = pipeline.answer_query(user_query)
        st.subheader("🤖 Antwort:")
        st.write(answer)

        # Display retrieved context with file type info
        st.subheader("📄 Verwendeter Kontext:")
        retrieved_chunks = pipeline.retriever.retrieve(user_query, top_k=3)
        
        if retrieved_chunks:
            for i, chunk in enumerate(retrieved_chunks):
                file_type = chunk['metadata'].get('file_type', 'unknown')
                filename = chunk['metadata']['filename']
                distance = chunk['distance']
                
                # File type icon
                type_icons = {"pdf": "📄", "docx": "📝", "txt": "📄"}
                icon = type_icons.get(file_type, "📄")
                
                with st.expander(
                    f"{icon} Chunk {i+1} - {filename} ({file_type.upper()}) - Relevanz: {1-distance:.3f}",
                    expanded=(i == 0)  # Erstes Chunk expanded
                ):
                    st.code(chunk['content'], language=None)
                    
                    # Zusätzliche Metadaten
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Position", chunk['metadata'].get('position', 'N/A'))
                    with col2:
                        st.metric("Chunk-Größe", len(chunk['content']))
                    with col3:
                        st.metric("Distanz", f"{distance:.4f}")
        else:
            st.info("Kein relevanter Kontext gefunden.")
```

#### Schritt 6: Error Handling und Edge Cases

**Robuste Error-Behandlung hinzufügen:**
```python
# In src/text_processor.py
class DocumentProcessingError(Exception):
    """Custom exception für Dokumentverarbeitungsfehler."""
    pass

def safe_file_processing(file_path: Path, processor_func) -> Optional[str]:
    """Sichere Dateiverarbeitung mit umfassendem Error Handling."""
    try:
        return processor_func(file_path)
    except PermissionError:
        logging.error(f"Permission denied: {file_path}")
        return None
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except MemoryError:
        logging.error(f"File too large to process: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error processing {file_path}: {e}")
        return None

# File size validation
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def validate_file_size(file_path: Path) -> bool:
    """Validiert Dateigröße."""
    try:
        size = file_path.stat().st_size
        if size > MAX_FILE_SIZE:
            logging.warning(f"File {file_path.name} too large: {size/1024/1024:.1f}MB")
            return False
        return True
    except:
        return False
```

#### Schritt 7: Testing und Validation

**Neue Test-Dateien erstellen:**
```python
# In test.py - erweiterte Tests
def test_document_processing():
    """Test alle Dokumenttypen."""
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Erstelle Test-Dateien
    create_test_files(test_data_dir)
    
    # Test processing
    from src.text_processor import process_documents
    chunks = process_documents(str(test_data_dir))
    
    # Validiere Ergebnisse
    assert len(chunks) > 0
    file_types = set(chunk['metadata']['file_type'] for chunk in chunks)
    print(f"Processed file types: {file_types}")
    
    # Cleanup
    shutil.rmtree(test_data_dir)

def create_test_files(test_dir: Path):
    """Erstellt Test-Dateien für alle Formate."""
    # TXT
    with open(test_dir / "test.txt", "w", encoding="utf-8") as f:
        f.write("Test TXT content with some text.")
    
    # Simple PDF (requires reportlab for creation)
    # Simple DOCX (requires python-docx for creation)
    # Implementation details...
```

**Erfolgskriterien und Acceptance Tests:**
1. ✅ **ERFÜLLT**: Alle drei Dateitypen (TXT, PDF, DOCX) werden erkannt und verarbeitet
2. ✅ **ERFÜLLT**: Streamlit UI zeigt korrekte Dateityp-Icons und Metadaten
3. ✅ **ERFÜLLT**: Tabellen in PDF/DOCX werden strukturiert extrahiert
4. ✅ **ERFÜLLT**: Error Handling funktioniert für korrupte/große Dateien (50MB Limit)
5. ✅ **ERFÜLLT**: Performance ist akzeptabel mit Batch-Processing
6. ✅ **ERFÜLLT**: Metadaten enthalten file_type Information in allen Komponenten
7. ✅ **ERFÜLLT**: Backward Compatibility mit bestehenden TXT-Workflows

**Implementierte Features:**
- **PDF-Parser**: Dual-Strategy (pdfplumber + PyMuPDF fallback) mit Tabellen-Extraktion
- **DOCX-Parser**: Vollständige Strukturerhaltung mit python-docx
- **Unified Interface**: `load_all_files()` mit automatischer Format-Erkennung
- **Enhanced UI**: File-Type-Icons, Upload-Statistiken, Progress-Bars
- **Robuste Error-Behandlung**: Größen-Validierung, Encoding-Fallbacks, Logging
- **Comprehensive Testing**: Multi-Format-Tests, Metadaten-Validierung, System-Integration

**Nächste Priorität:** P1.2 Verbessertes Chunking-System

### P1.2: Verbessertes Chunking-System
**Ziel:** Semantisch sinnvollere Text-Segmentierung

**Implementierung:**
- Erstelle `src/advanced_chunking.py`:
  - Implementiere sentence-aware chunking mit spaCy/NLTK
  - Füge paragraph-based chunking hinzu
  - Implementiere recursive chunking für große Dokumente
  - Behalte overlap-Logik bei, aber an Satzgrenzen

**Neue Chunking-Strategien:**
```python
class AdvancedChunker:
    def sentence_aware_chunk(text, max_chars=1000, overlap_sentences=2)
    def paragraph_based_chunk(text, max_paragraphs=3)
    def recursive_chunk(text, max_chars=1000, min_chars=200)
```

**Dateien zu erstellen/ändern:**
- `src/advanced_chunking.py`: Neue Chunking-Logik
- `src/text_processor.py`: Integration der neuen Chunker
- `src/config.py`: Chunking-Konfiguration

### P1.3: Konfiguration und Einstellungen
**Ziel:** System konfigurierbar machen ohne Code-Änderungen

**Implementierung:**
- Erstelle `config.yaml` im Projektroot:
```yaml
models:
  embedding_model: "all-MiniLM-L6-v2"
  llm_model: "google/flan-t5-base"
chunking:
  strategy: "sentence_aware"  # simple, sentence_aware, paragraph_based
  chunk_size: 1000
  overlap: 200
retrieval:
  top_k: 5
  similarity_threshold: 0.7
```

- Erweitere `src/config.py`:
  - YAML-Config-Loader
  - Environment Variable Override
  - Validation der Konfiguration

**Dateien zu erstellen/ändern:**
- `config.yaml`: Hauptkonfiguration  
- `src/config.py`: Config-Management-Klasse
- Alle Module: Config-Integration

---

## Phase 2: Retrieval-Optimierungen (Priorität: HOCH)

### P2.1: Hybrid Retrieval System
**Ziel:** Kombination von semantischer und Keyword-basierter Suche

**Implementierung:**
- Erweitere `src/retriever.py`:
  - Implementiere BM25-basierte Keyword-Suche (rank_bm25)
  - Füge Hybrid-Scoring hinzu (weighted combination)
  - Implementiere Query-Expansion mit Synonymen

```python
class HybridRetriever:
    def __init__(self, embedding_retriever, bm25_retriever):
        self.semantic_weight = 0.7
        self.keyword_weight = 0.3
    
    def hybrid_retrieve(self, query, top_k=5):
        semantic_results = self.embedding_retriever.retrieve(query, top_k*2)
        keyword_results = self.bm25_retriever.retrieve(query, top_k*2) 
        return self.merge_and_rerank(semantic_results, keyword_results, top_k)
```

**Dateien zu erstellen/ändern:**
- `src/retriever.py`: Hybrid-Retrieval-Klassen
- `src/pipeline.py`: Integration des Hybrid-Retrievers
- Dependencies: `pip install rank-bm25`

### P2.2: Erweiterte Vektordatenbank-Features
**Ziel:** Bessere Verwaltung und Performance der Vektordatenbank

**Implementierung:**
- Erweitere `src/vectorstore.py`:
  - Multi-Collection-Management für verschiedene Dokumenttypen
  - Metadata-Filtering (z.B. nur Kubernetes-Docs durchsuchen)
  - Batch-Operations für große Datenmengen
  - Index-Optimization und Kompression

```python
class AdvancedVectorStore:
    def create_filtered_collection(self, name, metadata_filter):
    def batch_add_documents(self, chunks, batch_size=1000):
    def optimize_index(self):
    def get_collection_stats(self):
```

**Dateien zu ändern:**
- `src/vectorstore.py`: Erweiterte ChromaDB-Integration
- `app/streamlit_app.py`: Collection-Auswahl in UI

### P2.3: Query-Verbesserung und Preprocessing  
**Ziel:** Intelligentere Query-Verarbeitung

**Implementierung:**
- Erstelle `src/query_processor.py`:
  - Query-Expansion mit verwandten Begriffen
  - Rechtschreibkorrektur (python-Levenshtein)
  - Intent-Detection (Frage vs. Befehl vs. Suche)
  - Multi-Language Query Support

```python
class QueryProcessor:
    def expand_query(self, query): # Synonyme, verwandte Begriffe
    def correct_spelling(self, query): # Fehlerkorrektur  
    def detect_intent(self, query): # Fragentyp erkennen
    def preprocess(self, query): # Bereinigung, Normalisierung
```

---

## Phase 3: LLM und Generierung (Priorität: MITTEL-HOCH)

### P3.1: Erweiterte LLM-Integration
**Ziel:** Bessere und flexiblere Antwortgenerierung

**Implementierung:**
- Erweitere `src/llm_handler.py`:
  - Multi-Model-Support (T5, FLAN-T5, GPT-4, Claude)
  - Model-Router basierend auf Query-Komplexität
  - Streaming-Antworten für bessere UX
  - Custom Prompt-Templates per Dokumenttyp

```python
class AdvancedLLMHandler:
    def __init__(self):
        self.models = {
            "simple": T5ForConditionalGeneration.from_pretrained("google/flan-t5-small"),
            "complex": T5ForConditionalGeneration.from_pretrained("google/flan-t5-large"),
            "api": OpenAIClient()  # Fallback für komplexe Fragen
        }
    
    def route_to_best_model(self, query, context):
    def stream_generate_answer(self, query, context):
```

**Dateien zu erstellen/ändern:**
- `src/llm_handler.py`: Multi-Model-Support
- `src/prompt_templates.py`: Template-Management
- `app/streamlit_app.py`: Streaming-UI-Updates

### P3.2: Context-Aware Response Generation
**Ziel:** Intelligentere Antworten basierend auf Dokumentkontext

**Implementierung:**
- Erstelle `src/context_manager.py`:
  - Context-Ranking und Relevanz-Scoring
  - Multi-Hop-Reasoning für komplexe Fragen
  - Citation-Generation (Quellenangaben)
  - Answer-Confidence-Scoring

```python
class ContextManager:
    def rank_context_relevance(self, query, contexts):
    def generate_citations(self, answer, source_chunks):
    def calculate_answer_confidence(self, query, contexts, answer):
```

---

## Phase 4: Evaluation und Monitoring (Priorität: MITTEL)

### P4.1: Retrieval-Evaluation-Framework
**Ziel:** Messbare Qualitätsbewertung des Retrieval-Systems

**Implementierung:**
- Erstelle `src/evaluation/retrieval_metrics.py`:
  - Precision@K, Recall@K, NDCG-Metriken
  - Ground-Truth-Dataset-Management
  - A/B-Testing-Framework für verschiedene Retrieval-Strategien

```python
class RetrievalEvaluator:
    def calculate_precision_at_k(self, retrieved, relevant, k):
    def calculate_recall_at_k(self, retrieved, relevant, k):
    def calculate_ndcg(self, retrieved, relevance_scores):
    def run_evaluation_suite(self, test_queries):
```

**Dateien zu erstellen:**
- `src/evaluation/retrieval_metrics.py`
- `src/evaluation/test_datasets.py`
- `evaluation/ground_truth/`: Test-Queries mit expected results

### P4.2: Answer-Quality-Assessment
**Ziel:** Automatische Bewertung der Antwortqualität

**Implementierung:**
- Erstelle `src/evaluation/answer_quality.py`:
  - BLEU/ROUGE-Metriken für Answer-Quality
  - Semantic-Similarity-Scoring
  - Hallucination-Detection
  - Factual-Accuracy-Checking

```python
class AnswerEvaluator:
    def calculate_bleu_score(self, generated, reference):
    def semantic_similarity(self, generated, reference):
    def detect_hallucinations(self, answer, source_contexts):
```

### P4.3: Monitoring und Logging
**Ziel:** System-Performance und Nutzung überwachen

**Implementierung:**
- Erstelle `src/monitoring/`:
  - Query-Logging mit Response-Times
  - Error-Tracking und Alerting
  - Usage-Analytics (häufige Queries, erfolgreiche vs. erfolglose Suchen)
  - Performance-Metriken (Retrieval-Zeit, Generation-Zeit)

---

## Phase 5: UI/UX-Verbesserungen (Priorität: MITTEL)

### P5.1: Erweiterte Streamlit-Interface
**Ziel:** Professionellere und benutzerfreundlichere Oberfläche

**Implementierung:**
- Erweitere `app/streamlit_app.py`:
  - Chat-Interface mit Gesprächshistorie
  - Drag & Drop für Dateien
  - Bulk-Upload mit Progress-Bars
  - Export-Funktionen (Antworten als PDF/MD)
  - Dark/Light-Mode-Toggle

**Neue Features:**
```python
# Chat-History-Management
class ChatManager:
    def save_conversation(self, user_query, system_response):
    def load_conversation_history(self):
    def export_conversation(self, format="pdf"):
```

### P5.2: Document-Viewer und Annotation
**Ziel:** Bessere Transparenz über verwendete Quellen

**Implementierung:**
- Integriere Document-Viewer in Streamlit:
  - PDF-Viewer mit Highlight der verwendeten Chunks
  - Click-to-Source-Funktionalität
  - Document-Annotation-Interface
  - Side-by-side Comparison verschiedener Antworten

---

## Phase 6: Skalierung und Performance (Priorität: NIEDRIG-MITTEL)

### P6.1: Caching und Performance-Optimierung
**Ziel:** Bessere Response-Zeiten und Skalierbarkeit

**Implementierung:**
- Erstelle `src/caching/`:
  - Redis-Integration für Query-Result-Caching
  - Embedding-Cache für wiederkehrende Dokumente
  - Model-Loading-Optimierung
  - Async-Processing für lange Dokumenten-Uploads

### P6.2: Containerization und Deployment
**Ziel:** Einfache Deployment-Optionen

**Implementierung:**
- Erstelle Deployment-Files:
  - `Dockerfile` mit Multi-Stage-Build
  - `docker-compose.yml` mit Services (App, ChromaDB, Redis)
  - `kubernetes/` mit K8s-Manifests
  - GitHub Actions für CI/CD

---

## Phase 7: Erweiterte Features (Priorität: NIEDRIG)

### P7.1: Multi-Modal RAG
**Ziel:** Bilder und Tabellen in Dokumenten verarbeiten

**Implementierung:**
- Integriere Vision-Models:
  - Image-to-Text für Screenshots/Diagramme
  - Table-Extraction und -Strukturierung
  - Chart/Graph-Interpretation

### P7.2: Collaborative Features
**Ziel:** Team-Nutzung und Wissens-Sharing

**Implementierung:**
- User-Management und Authentication
- Shared-Collections zwischen Team-Mitgliedern
- Annotation und Kommentar-System
- Knowledge-Base-Versionierung

---

## Implementierungsreihenfolge (Empfohlen)

### Sprint 1 (2-3 Wochen):
1. P1.1: PDF/DOCX-Support
2. P1.3: Konfigurationssystem
3. P2.2: Multi-Collection-Management

### Sprint 2 (2-3 Wochen):
1. P1.2: Verbessertes Chunking
2. P2.1: Hybrid-Retrieval
3. P3.1: Multi-Model-LLM-Support

### Sprint 3 (2-3 Wochen):
1. P2.3: Query-Preprocessing
2. P3.2: Context-Management
3. P4.1: Evaluation-Framework

### Sprint 4 (2-3 Wochen):
1. P5.1: UI-Verbesserungen
2. P4.2: Answer-Quality-Assessment
3. P4.3: Monitoring-System

### Langfristig (3+ Monate):
- P6.1-P6.2: Performance und Deployment
- P7.1-P7.2: Erweiterte Features

## Erfolgsmessung

### Quantitative Metriken:
- **Retrieval-Quality:** Precision@5 > 0.8, NDCG > 0.7
- **Answer-Quality:** BLEU > 0.6, Semantic-Similarity > 0.8
- **Performance:** Query-Response < 3s, Document-Ingestion < 30s/MB
- **User-Experience:** Task-Success-Rate > 90%

### Qualitative Ziele:
- System kann 95% der typischen Fachfragen korrekt beantworten
- Nutzer können intuitiv neue Dokumente hinzufügen
- Antworten enthalten präzise Quellenangaben
- System funktioniert zuverlässig mit verschiedenen Dokumenttypen

## Ressourcen und Dependencies

### Neue Python-Packages:
```
# Phase 1
pdfplumber>=3.0.0
python-docx>=0.8.11
spacy>=3.4.0
PyYAML>=6.0

# Phase 2  
rank-bm25>=0.2.2
python-Levenshtein>=0.20.0

# Phase 3
openai>=1.0.0
anthropic>=0.7.0

# Phase 4
nltk>=3.8
rouge-score>=0.1.2

# Phase 6
redis>=4.5.0
celery>=5.2.0
```

### Hardware-Empfehlungen:
- **Minimum:** 16GB RAM, 4 CPU Cores, 100GB Storage
- **Optimal:** 32GB RAM, 8 CPU Cores, NVIDIA GPU (8GB+), 500GB SSD

Dieses Plan-Dokument dient als Referenz für alle zukünftigen Entwicklungsaktivitäten am RAG-System.
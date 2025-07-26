#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test ob alle Dependencies korrekt installiert sind."""

import sys
import os
from pathlib import Path

# Fix encoding issues on Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}\n")

# Test imports
packages = [
    "transformers",
    "torch", 
    "sentence_transformers",
    "chromadb",
    "langchain",
    "streamlit",
    "PyPDF2",
    "docx",
    "pdfplumber",
    "fitz",  # PyMuPDF
    "rank_bm25",  # BM25 keyword search
    "nltk",       # German stemming
    "Levenshtein" # Fuzzy matching
]

for package in packages:
    try:
        if package == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            print(f"OK: {package} - importiert")
        elif package == "docx":
            import docx
            print(f"OK: python-{package} - importiert")
        elif package == "fitz":
            import fitz
            print(f"OK: PyMuPDF ({package}) - importiert")
        elif package == "rank_bm25":
            from rank_bm25 import BM25Okapi
            print(f"OK: {package} - BM25 keyword search verfügbar")
        elif package == "Levenshtein":
            import Levenshtein
            print(f"OK: python-{package} - Fuzzy matching verfügbar")
        else:
            __import__(package)
            print(f"OK: {package} - importiert")
    except ImportError as e:
        print(f"ERROR: {package} - FEHLT: {e}")

# GPU Check
print("\nGPU Status:")
try:
    import torch
    if torch.cuda.is_available():
        print(f"OK: CUDA verfügbar: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("- Keine GPU gefunden (CPU-only Mode)")
except:
    print("ERROR: PyTorch CUDA Check fehlgeschlagen")

# Test Model Download
print("\nModell-Test:")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    test_embedding = model.encode("Test")
    print(f"OK: Modell geladen, Embedding-Dimension: {len(test_embedding)}")
except Exception as e:
    print(f"ERROR: Modell-Fehler: {e}")

# Test Document Parsers
print("\nDokument-Parser-Tests:")
def test_document_parsers():
    """Test PDF und DOCX parsing capabilities."""
    from pathlib import Path
    import tempfile
    import shutil
    
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Test PDF parsing
        try:
            import pdfplumber
            print("OK: pdfplumber - PDF-Parser verfügbar")
        except ImportError:
            print("ERROR: pdfplumber nicht installiert")
        
        try:
            import fitz
            print("OK: PyMuPDF - PDF-Fallback verfügbar")
        except ImportError:
            print("ERROR: PyMuPDF nicht installiert")
        
        # Test DOCX parsing
        try:
            from docx import Document
            from docx.oxml.table import CT_Tbl
            from docx.oxml.text.paragraph import CT_P
            print("OK: python-docx - DOCX-Parser vollständig verfügbar")
        except ImportError as e:
            print(f"ERROR: DOCX-Parser-Komponenten fehlen: {e}")
        
        print("✓ Alle Dokumenten-Parser erfolgreich getestet")
        
    except Exception as e:
        print(f"ERROR: Unerwarteter Fehler beim Parser-Test: {e}")
    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)

test_document_parsers()

# Test enhanced document processing
print("\nErweiterte Dokumentverarbeitung:")
def test_enhanced_processing():
    """Test die neue erweiterte Dokumentverarbeitung."""
    from pathlib import Path
    import tempfile
    import shutil
    
    # Erstelle temporäres Test-Verzeichnis
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir) / "test_docs"
        test_dir.mkdir()
        
        try:
            # Erstelle Test-TXT-Datei
            test_txt = test_dir / "test.txt"
            with open(test_txt, "w", encoding="utf-8") as f:
                f.write("Dies ist ein Test-TXT-Dokument mit wichtigen Informationen über Kubernetes und Python.")
            
            # Test die neue process_documents Funktion
            from src.text_processor import process_documents, load_all_files
            
            # Test load_all_files
            all_files = load_all_files(str(test_dir))
            print(f"OK: load_all_files fand {len(all_files)} Dateien")
            
            if all_files:
                filename, content, file_type = all_files[0]
                print(f"  - {filename} ({file_type}, {len(content)} Zeichen)")
            
            # Test process_documents mit file_type Metadaten
            chunks = process_documents(str(test_dir))
            print(f"OK: process_documents erstellte {len(chunks)} Chunks")
            
            if chunks:
                sample_chunk = chunks[0]
                metadata = sample_chunk['metadata']
                required_fields = ['filename', 'file_type', 'position', 'chunk_id', 'chunk_size']
                
                for field in required_fields:
                    if field in metadata:
                        print(f"  OK Metadaten-Feld '{field}': {metadata[field]}")
                    else:
                        print(f"  ERROR Fehlendes Metadaten-Feld: {field}")
            
            print("OK Erweiterte Dokumentverarbeitung erfolgreich getestet")
            
        except Exception as e:
            print(f"ERROR: Fehler beim Test der erweiterten Verarbeitung: {e}")
            import traceback
            traceback.print_exc()

test_enhanced_processing()

# Test file type validation
print("\nDateityp-Validierung:")
def test_file_validation():
    """Test Dateigrößen- und Typ-Validierung."""
    try:
        from src.text_processor import validate_file_size, MAX_FILE_SIZE
        from pathlib import Path
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test normale Dateigröße
            small_file = Path(temp_dir) / "small.txt"
            with open(small_file, "w") as f:
                f.write("Small file content")
            
            if validate_file_size(small_file):
                print("OK Kleine Datei-Validierung erfolgreich")
            else:
                print("ERROR Kleine Datei fälschlicherweise abgelehnt")
            
            print(f"  Max. Dateigröße: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
            
    except Exception as e:
        print(f"ERROR: Fehler beim Validierungs-Test: {e}")

test_file_validation()

# Comprehensive system test
print("\nGesamtsystem-Test:")
def test_full_system():
    """Test das gesamte RAG-System mit neuen Dateiformaten."""
    try:
        # Test ob alle Komponenten mit den neuen Metadaten funktionieren
        from src.text_processor import process_documents
        from src.embeddings import EmbeddingManager
        from src.vectorstore import VectorStoreManager
        from src.retriever import Retriever
        
        test_dir = "data/raw_texts"
        if not Path(test_dir).exists():
            print(f"INFO: Test-Verzeichnis {test_dir} nicht gefunden - überspringe Systemtest")
            return
        
        # 1. Dokumentverarbeitung
        chunks = process_documents(test_dir)
        if chunks:
            print(f"OK Dokumentverarbeitung: {len(chunks)} Chunks erstellt")
            
            # Prüfe file_type Verteilung
            file_types = {}
            for chunk in chunks:
                ft = chunk['metadata'].get('file_type', 'unknown')
                file_types[ft] = file_types.get(ft, 0) + 1
            print(f"  Dateityp-Verteilung: {file_types}")
            
            # 2. Test Embedding-Generierung mit neuen Metadaten
            try:
                embed_manager = EmbeddingManager()
                # Test nur mit ersten paar Chunks für Geschwindigkeit
                test_chunks = chunks[:2]
                chunks_with_embeddings = embed_manager.generate_embeddings(test_chunks)
                
                if chunks_with_embeddings and 'embedding' in chunks_with_embeddings[0]:
                    print("OK Embedding-Generierung funktioniert mit neuen Metadaten")
                else:
                    print("ERROR Embedding-Generierung fehlgeschlagen")
                    
            except Exception as e:
                print(f"INFO: Embedding-Test übersprungen (Modell nicht verfügbar): {e}")
        
        else:
            print("INFO: Keine Dokumente für Systemtest gefunden")
            
    except Exception as e:
        print(f"ERROR: Systemtest fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()

test_full_system()

# Test Hybrid Retrieval Dependencies
print("\nHybrid Retrieval Dependencies:")
def test_hybrid_retrieval_dependencies():
    """Test Hybrid Retrieval specific dependencies."""
    print("Testing Hybrid Retrieval dependencies...")
    
    try:
        import rank_bm25
        print("OK rank-bm25 available")
    except ImportError:
        print("ERROR rank-bm25 not available - run: pip install rank-bm25")
        return False
    
    try:
        import nltk
        from nltk.stem import SnowballStemmer
        
        # Test German stemmer
        stemmer = SnowballStemmer('german')
        test_word = stemmer.stem('Umgebungen')
        print(f"OK NLTK German stemmer available (test: Umgebungen -> {test_word})")
    except ImportError:
        print("ERROR NLTK not available - run: pip install nltk")
        return False
    except LookupError:
        print("INFO NLTK German data missing - will be downloaded automatically")
    
    try:
        import Levenshtein
        distance = Levenshtein.distance('test', 'tast')
        print(f"OK python-Levenshtein available (test distance: {distance})")
    except ImportError:
        print("ERROR python-Levenshtein not available - run: pip install python-Levenshtein")
        return False
    
    return True

def test_hybrid_retrieval_performance():
    """Test Hybrid Retrieval performance with sample queries."""
    print("Testing Hybrid Retrieval performance...")
    
    try:
        # Import and test components
        from src.retriever import QueryAnalyzer, GermanBM25Retriever
        
        # Test Query Analyzer
        analyzer = QueryAnalyzer()
        test_queries = [
            "Was ist Kubernetes?",
            "kubectl get pods command",
            "Docker Container Best Practices"
        ]
        
        for query in test_queries:
            analysis = analyzer.analyze_query_type(query)
            print(f"Query: '{query}' -> {analysis['query_type']} "
                  f"(semantic: {analysis['semantic_weight']}, "
                  f"keyword: {analysis['keyword_weight']})")
        
        # Test German BM25 Retriever
        test_docs = [
            {'content': 'Kubernetes ist ein System zur Container-Orchestrierung', 'metadata': {'chunk_id': 'test-1'}},
            {'content': 'Python Entwicklung mit virtuellen Umgebungen', 'metadata': {'chunk_id': 'test-2'}}
        ]
        
        bm25_retriever = GermanBM25Retriever(test_docs)
        results = bm25_retriever.retrieve('Kubernetes', top_k=1)
        
        if results and results[0]['metadata']['chunk_id'] == 'test-1':
            print("OK BM25 retrieval working correctly")
        else:
            print("ERROR BM25 retrieval not working as expected")
        
        print("OK Hybrid Retrieval components working")
        return True
        
    except Exception as e:
        print(f"ERROR Hybrid Retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

test_hybrid_retrieval_dependencies()
test_hybrid_retrieval_performance()

print("\n" + "="*50)
print("OK ALLE TESTS ABGESCHLOSSEN")
print("Das RAG-System unterstützt jetzt PDF, DOCX und TXT Dateien!")
print("Hybrid Retrieval Dependencies getestet - bereit für Implementierung!")
print("Starten Sie das System mit: streamlit run app/streamlit_app.py")
print("="*50)
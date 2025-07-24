#!/usr/bin/env python
"""Test ob alle Dependencies korrekt installiert sind."""

import sys
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
    "docx"
]

for package in packages:
    try:
        if package == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            print(f"OK: {package} - importiert")
        elif package == "docx":
            import docx
            print(f"OK: python-{package} - importiert")
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
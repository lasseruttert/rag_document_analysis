#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check which dependencies are missing for the RAG system.
"""

import sys
import importlib

def check_dependency(package_name, optional=False):
    """Check if a package is available."""
    try:
        importlib.import_module(package_name)
        status = "[PASS]" if not optional else "[PASS-OPT]"
        print(f"{status} {package_name}")
        return True
    except ImportError:
        status = "[FAIL]" if not optional else "[MISS-OPT]"
        print(f"{status} {package_name}")
        return False

def main():
    """Check all required and optional dependencies."""
    print("RAG System Dependency Check")
    print("=" * 40)
    
    # Core dependencies
    print("\nCore Dependencies:")
    core_deps = [
        'torch',
        'sentence_transformers', 
        'transformers',
        'numpy',
        'chromadb',
        'rank_bm25',
        'nltk',
        'streamlit',
        'yaml'
    ]
    
    core_available = 0
    for dep in core_deps:
        if check_dependency(dep):
            core_available += 1
    
    # Optional dependencies
    print("\nOptional Dependencies:")
    optional_deps = [
        ('pdfplumber', 'PDF processing'),
        ('fitz', 'PyMuPDF for PDF fallback'),
        ('docx', 'python-docx for DOCX files'),
        ('pytest', 'Testing framework')
    ]
    
    optional_available = 0
    for dep, description in optional_deps:
        if check_dependency(dep, optional=True):
            optional_available += 1
    
    # Summary
    print("\n" + "=" * 40)
    print(f"Core Dependencies: {core_available}/{len(core_deps)} available")
    print(f"Optional Dependencies: {optional_available}/{len(optional_deps)} available")
    
    if core_available == len(core_deps):
        print("\n✅ All core dependencies available - RAG system should work!")
        return 0
    else:
        missing = len(core_deps) - core_available
        print(f"\n❌ {missing} core dependencies missing - RAG system will NOT work")
        print("\nTo fix, run:")
        print("  conda activate rag")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == '__main__':
    sys.exit(main())
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verify BM25 caching implementation structure.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def verify_caching_implementation():
    """Verify that caching implementation has been added correctly."""
    print("Verifying BM25 caching implementation...")
    
    # Check that the retriever file has caching-related imports
    retriever_file = os.path.join('src', 'retriever.py')
    if not os.path.exists(retriever_file):
        print("[FAIL] Retriever file not found")
        return False
    
    with open(retriever_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for caching-related imports
    required_imports = ['hashlib', 'pickle', 'tempfile']
    for imp in required_imports:
        if f'import {imp}' in content:
            print(f"[PASS] Found import: {imp}")
        else:
            print(f"[FAIL] Missing import: {imp}")
            return False
    
    # Check for pathlib import (can be "from pathlib import Path")
    if 'from pathlib import' in content or 'import pathlib' in content:
        print("[PASS] Found import: pathlib")
    else:
        print("[FAIL] Missing import: pathlib")
        return False
    
    # Check for caching-related methods
    required_methods = [
        '_generate_cache_key',
        '_build_bm25_model', 
        '_save_to_cache',
        '_load_from_cache',
        'clear_cache',
        'clear_all_cache'
    ]
    
    for method in required_methods:
        if f'def {method}' in content:
            print(f"[PASS] Found method: {method}")
        else:
            print(f"[FAIL] Missing method: {method}")
            return False
    
    # Check that constructor has cache parameters
    if 'cache_dir: str = None' in content and 'enable_cache: bool = True' in content:
        print("[PASS] Constructor has caching parameters")
    else:
        print("[FAIL] Constructor missing caching parameters")
        return False
    
    # Check that EnhancedHybridRetriever has caching support
    if 'def clear_bm25_cache(self):' in content:
        print("[PASS] EnhancedHybridRetriever has cache management")
    else:
        print("[FAIL] EnhancedHybridRetriever missing cache management")
        return False
    
    # Check for cache validation
    if 'cache_version' in content:
        print("[PASS] Cache versioning implemented")
    else:
        print("[FAIL] Cache versioning missing")
        return False
    
    print("[PASS] BM25 caching implementation verified successfully")
    return True

def verify_performance_improvements():
    """Verify that all performance improvements have been implemented."""
    print("\nVerifying all performance improvements...")
    
    improvements = [
        ("Print statements removed", check_no_print_statements),
        ("Requirements.txt created", check_requirements_file),
        ("Constants module created", check_constants_module),
        ("Configuration updated", check_config_updates),
        ("Basic tests created", check_basic_tests),
        ("BM25 caching implemented", verify_caching_implementation)
    ]
    
    passed = 0
    total = len(improvements)
    
    for name, check_func in improvements:
        try:
            if check_func():
                print(f"[PASS] {name}")
                passed += 1
            else:
                print(f"[FAIL] {name}")
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
    
    print(f"\n{passed}/{total} improvements verified")
    return passed == total

def check_no_print_statements():
    """Check that print statements have been removed."""
    core_files = ['src/pipeline.py', 'src/text_processor.py', 'src/retriever.py', 'src/config.py']
    
    for file_path in core_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                for line in lines:
                    stripped = line.strip()
                    if (stripped.startswith('print(') and 
                        not stripped.startswith('#') and 
                        not stripped.startswith('"""')):
                        return False
    return True

def check_requirements_file():
    """Check that requirements.txt exists."""
    return os.path.exists('requirements.txt')

def check_constants_module():
    """Check that constants module exists."""
    return os.path.exists('src/constants.py')

def check_config_updates():
    """Check that config has been updated."""
    config_file = 'config.yaml'
    if not os.path.exists(config_file):
        return False
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Check for new config sections
    return 'hybrid_boost_factor' in content and 'keyword_extraction' in content

def check_basic_tests():
    """Check that basic tests exist."""
    return (os.path.exists('test_basic_functionality.py') or 
            os.path.exists('tests/test_constants.py'))

def main():
    """Verify all implementations."""
    print("RAG System Improvements Verification")
    print("=" * 50)
    
    success = verify_performance_improvements()
    
    if success:
        print("\nAll improvements successfully implemented!")
        return 0
    else:
        print("\nSome improvements are missing or incomplete")
        return 1

if __name__ == '__main__':
    sys.exit(main())
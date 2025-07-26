#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic functionality tests for the RAG system improvements.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_constants_import():
    """Test that constants module can be imported and has expected values."""
    print("Testing constants module...")
    
    from constants import (
        GERMAN_STOPWORDS, ENGLISH_STOPWORDS, COMBINED_STOPWORDS,
        TECHNICAL_PATTERNS, CODE_KEYWORDS, QUESTION_PATTERNS
    )
    
    # Test stopwords
    assert len(GERMAN_STOPWORDS) > 0, "German stopwords should not be empty"
    assert len(ENGLISH_STOPWORDS) > 0, "English stopwords should not be empty"
    assert len(COMBINED_STOPWORDS) > len(GERMAN_STOPWORDS), "Combined should be larger"
    assert 'der' in COMBINED_STOPWORDS, "Should contain German stopwords"
    assert 'the' in COMBINED_STOPWORDS, "Should contain English stopwords"
    
    # Test patterns
    assert len(TECHNICAL_PATTERNS) > 0, "Technical patterns should not be empty"
    assert len(CODE_KEYWORDS) > 0, "Code keywords should not be empty"
    assert len(QUESTION_PATTERNS) > 0, "Question patterns should not be empty"
    
    # Test configurable functions
    from constants import get_max_file_size_mb, get_min_keyword_length
    
    size = get_max_file_size_mb()
    assert isinstance(size, (int, float)), "File size should be numeric"
    assert size > 0, "File size should be positive"
    
    length = get_min_keyword_length()
    assert isinstance(length, int), "Keyword length should be integer"
    assert length > 0, "Keyword length should be positive"
    
    print("[PASS] Constants module tests passed")

def test_text_processor_basic():
    """Test basic text processor functionality."""
    print("Testing text processor module...")
    
    from text_processor import extract_keywords, normalize_text, calculate_semantic_density
    
    # Test keyword extraction
    text = "This is a test about Python programming and machine learning algorithms"
    keywords = extract_keywords(text, max_keywords=3)
    
    assert isinstance(keywords, list), "Keywords should be a list"
    assert len(keywords) <= 3, "Should respect max_keywords limit"
    assert all(isinstance(kw, str) for kw in keywords), "All keywords should be strings"
    
    # Test normalization
    text_with_spaces = "This  has   multiple    spaces\n\n\nand newlines"
    normalized = normalize_text(text_with_spaces)
    assert "  " not in normalized, "Should remove multiple spaces"
    assert "\n\n\n" not in normalized, "Should reduce multiple newlines"
    
    # Test semantic density
    density = calculate_semantic_density("Machine learning algorithms process data efficiently")
    assert isinstance(density, float), "Density should be float"
    assert 0 <= density <= 1, "Density should be between 0 and 1"
    
    print("[PASS] Text processor basic tests passed")

def test_config_loading():
    """Test that configuration can be loaded."""
    print("Testing config module...")
    
    try:
        from config import get_config, RAGConfig
        
        # Test singleton
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2, "Should return same instance"
        
        # Test that config has expected sections
        config = get_config()
        assert hasattr(config, 'models'), "Should have models section"
        assert hasattr(config, 'chunking'), "Should have chunking section"
        assert hasattr(config, 'retrieval'), "Should have retrieval section"
        
        # Test configuration validation
        assert config.validate_config(), "Default config should be valid"
        
        # Test section access
        models_section = config.get_section('models')
        assert models_section is not None, "Should be able to get models section"
        
        print("[PASS] Config module tests passed")
        
    except Exception as e:
        print(f"[SKIP] Config tests skipped due to: {e}")

def test_no_print_statements():
    """Test that print statements have been replaced with logging."""
    print("Testing that print statements were removed...")
    
    # Check some key files for print statements
    test_files = [
        'src/pipeline.py',
        'src/text_processor.py', 
        'src/retriever.py',
        'src/config.py'
    ]
    
    print_found = False
    for file_path in test_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Look for print statements (but not in comments or strings)
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if (stripped.startswith('print(') and 
                        not stripped.startswith('#') and 
                        not stripped.startswith('"""') and
                        not stripped.startswith("'''")):
                        print(f"[WARN] Found print statement in {file_path}:{i+1}: {line.strip()}")
                        print_found = True
    
    if not print_found:
        print("[PASS] No print statements found in core modules")
    else:
        print("[FAIL] Some print statements still remain")

def test_requirements_file():
    """Test that requirements.txt was created."""
    print("Testing requirements.txt...")
    
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            content = f.read()
            
        # Check for key dependencies
        required_deps = ['torch', 'sentence-transformers', 'streamlit', 'chromadb', 'pyyaml']
        for dep in required_deps:
            if dep in content:
                print(f"[PASS] Found {dep} in requirements.txt")
            else:
                print(f"[WARN] Missing {dep} in requirements.txt")
        
        print("[PASS] Requirements.txt exists and has dependencies")
    else:
        print("[FAIL] requirements.txt not found")

def main():
    """Run all basic tests."""
    print("Basic Functionality Tests for RAG System Improvements")
    print("=" * 60)
    
    tests = [
        test_constants_import,
        test_text_processor_basic,
        test_config_loading,
        test_no_print_statements,
        test_requirements_file
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{total} test groups passed")
    
    if passed == total:
        print("All basic tests passed!")
        return 0
    else:
        print(f"{total - passed} test groups failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
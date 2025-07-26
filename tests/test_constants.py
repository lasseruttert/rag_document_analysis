#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for constants module.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from constants import (
    GERMAN_STOPWORDS, ENGLISH_STOPWORDS, COMBINED_STOPWORDS,
    TECHNICAL_PATTERNS, CODE_KEYWORDS, QUESTION_PATTERNS,
    get_max_file_size_mb, get_min_keyword_length, get_default_semantic_weight
)


class TestStopwords:
    """Test stopwords functionality."""
    
    def test_german_stopwords_not_empty(self):
        """Test that German stopwords set is not empty."""
        assert len(GERMAN_STOPWORDS) > 0
        assert 'der' in GERMAN_STOPWORDS
        assert 'die' in GERMAN_STOPWORDS
        assert 'und' in GERMAN_STOPWORDS
    
    def test_english_stopwords_not_empty(self):
        """Test that English stopwords set is not empty."""
        assert len(ENGLISH_STOPWORDS) > 0
        assert 'the' in ENGLISH_STOPWORDS
        assert 'and' in ENGLISH_STOPWORDS
        assert 'you' in ENGLISH_STOPWORDS
    
    def test_combined_stopwords(self):
        """Test that combined stopwords includes both German and English."""
        assert len(COMBINED_STOPWORDS) > len(GERMAN_STOPWORDS)
        assert len(COMBINED_STOPWORDS) > len(ENGLISH_STOPWORDS)
        assert 'der' in COMBINED_STOPWORDS
        assert 'the' in COMBINED_STOPWORDS
    
    def test_stopwords_are_sets(self):
        """Test that stopwords are implemented as sets for fast lookup."""
        assert isinstance(GERMAN_STOPWORDS, set)
        assert isinstance(ENGLISH_STOPWORDS, set)
        assert isinstance(COMBINED_STOPWORDS, set)


class TestPatterns:
    """Test pattern definitions."""
    
    def test_technical_patterns_list(self):
        """Test that technical patterns are defined."""
        assert isinstance(TECHNICAL_PATTERNS, list)
        assert len(TECHNICAL_PATTERNS) > 0
        # Test that patterns are strings that can be compiled as regex
        import re
        for pattern in TECHNICAL_PATTERNS:
            assert isinstance(pattern, str)
            re.compile(pattern)  # Should not raise exception
    
    def test_code_keywords_set(self):
        """Test that code keywords are defined."""
        assert isinstance(CODE_KEYWORDS, set)
        assert len(CODE_KEYWORDS) > 0
        assert 'python' in CODE_KEYWORDS
        assert 'kubernetes' in CODE_KEYWORDS
    
    def test_question_patterns_list(self):
        """Test that question patterns are defined."""
        assert isinstance(QUESTION_PATTERNS, list)
        assert len(QUESTION_PATTERNS) > 0
        import re
        for pattern in QUESTION_PATTERNS:
            assert isinstance(pattern, str)
            re.compile(pattern)  # Should not raise exception


class TestConfigurableConstants:
    """Test configurable constants functionality."""
    
    def test_get_max_file_size_mb_default(self):
        """Test that max file size returns a reasonable default."""
        size = get_max_file_size_mb()
        assert isinstance(size, (int, float))
        assert size > 0
        assert size <= 1000  # Reasonable upper bound
    
    def test_get_min_keyword_length_default(self):
        """Test that min keyword length returns a reasonable default."""
        length = get_min_keyword_length()
        assert isinstance(length, int)
        assert length >= 1
        assert length <= 10  # Reasonable upper bound
    
    def test_get_default_semantic_weight_default(self):
        """Test that semantic weight returns a value between 0 and 1."""
        weight = get_default_semantic_weight()
        assert isinstance(weight, (int, float))
        assert 0 <= weight <= 1


class TestBackwardCompatibility:
    """Test that legacy constants are still available."""
    
    def test_legacy_constants_exist(self):
        """Test that legacy constant names still work."""
        from constants import (
            DEFAULT_MAX_FILE_SIZE_MB, SUPPORTED_FILE_FORMATS,
            MIN_KEYWORD_LENGTH, MAX_KEYWORDS_DEFAULT,
            DEFAULT_SEMANTIC_WEIGHT, DEFAULT_KEYWORD_WEIGHT
        )
        
        # Test that they have reasonable values
        assert isinstance(DEFAULT_MAX_FILE_SIZE_MB, (int, float))
        assert isinstance(SUPPORTED_FILE_FORMATS, list)
        assert isinstance(MIN_KEYWORD_LENGTH, int)
        assert isinstance(MAX_KEYWORDS_DEFAULT, int)
        assert isinstance(DEFAULT_SEMANTIC_WEIGHT, (int, float))
        assert isinstance(DEFAULT_KEYWORD_WEIGHT, (int, float))
        
        # Test semantic + keyword weights sum to 1
        assert abs((DEFAULT_SEMANTIC_WEIGHT + DEFAULT_KEYWORD_WEIGHT) - 1.0) < 0.01


if __name__ == '__main__':
    pytest.main([__file__])
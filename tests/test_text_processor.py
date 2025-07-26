#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for text_processor module.
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from text_processor import (
    extract_keywords, normalize_text, split_into_sentences,
    calculate_semantic_density, chunk_text, split_large_sentence,
    is_heading
)


class TestExtractKeywords:
    """Test keyword extraction functionality."""
    
    def test_extract_keywords_basic(self):
        """Test basic keyword extraction."""
        text = "This is a simple test about Python programming and machine learning algorithms."
        keywords = extract_keywords(text, max_keywords=3)
        
        assert isinstance(keywords, list)
        assert len(keywords) <= 3
        assert all(isinstance(kw, str) for kw in keywords)
        # Should not contain common stopwords
        assert 'this' not in [kw.lower() for kw in keywords]
        assert 'and' not in [kw.lower() for kw in keywords]
    
    def test_extract_keywords_empty_text(self):
        """Test keyword extraction with empty text."""
        keywords = extract_keywords("")
        assert keywords == []
        
        keywords = extract_keywords("   ")
        assert keywords == []
    
    def test_extract_keywords_short_text(self):
        """Test keyword extraction with very short text."""
        text = "Hi there"
        keywords = extract_keywords(text)
        assert isinstance(keywords, list)
        # Short text might not have meaningful keywords
    
    def test_extract_keywords_technical_text(self):
        """Test keyword extraction with technical terms."""
        text = "Kubernetes pods deployment containers orchestration Docker microservices"
        keywords = extract_keywords(text, max_keywords=5)
        
        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        # Should prefer longer, more specific terms
        technical_terms = [kw.lower() for kw in keywords]
        assert any(term in ['kubernetes', 'deployment', 'containers', 'orchestration', 'microservices'] 
                  for term in technical_terms)


class TestNormalizeText:
    """Test text normalization functionality."""
    
    def test_normalize_text_basic(self):
        """Test basic text normalization."""
        text = "This  is   a    test\n\n\nwith   multiple    spaces"
        normalized = normalize_text(text)
        
        assert "  " not in normalized  # No double spaces
        assert "\n\n\n" not in normalized  # No triple newlines
        assert normalized.strip() == normalized  # No leading/trailing whitespace
    
    def test_normalize_text_empty(self):
        """Test normalization with empty text."""
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""
    
    def test_normalize_text_preserves_structure(self):
        """Test that normalization preserves paragraph structure."""
        text = "Paragraph 1\n\nParagraph 2\n\n\n\nParagraph 3"
        normalized = normalize_text(text)
        
        # Should have exactly one empty line between paragraphs
        assert "Paragraph 1\n\nParagraph 2\n\nParagraph 3" == normalized


class TestSplitIntoSentences:
    """Test sentence splitting functionality."""
    
    def test_split_sentences_basic(self):
        """Test basic sentence splitting."""
        text = "This is sentence one. This is sentence two! And this is sentence three?"
        sentences = split_into_sentences(text)
        
        assert len(sentences) == 3
        assert all(isinstance(s, str) for s in sentences)
        assert "This is sentence one." in sentences[0]
        assert "This is sentence two!" in sentences[1]
        assert "And this is sentence three?" in sentences[2]
    
    def test_split_sentences_empty(self):
        """Test sentence splitting with empty text."""
        sentences = split_into_sentences("")
        assert sentences == [""]
    
    def test_split_sentences_no_punctuation(self):
        """Test sentence splitting with text without punctuation."""
        text = "This is just one long sentence without any punctuation marks"
        sentences = split_into_sentences(text)
        
        assert len(sentences) >= 1
        assert text in sentences[0]


class TestCalculateSemanticDensity:
    """Test semantic density calculation."""
    
    def test_semantic_density_basic(self):
        """Test basic semantic density calculation."""
        text = "Machine learning algorithms process data efficiently using neural networks"
        density = calculate_semantic_density(text)
        
        assert isinstance(density, float)
        assert 0 <= density <= 1
    
    def test_semantic_density_empty(self):
        """Test semantic density with empty text."""
        assert calculate_semantic_density("") == 0.0
        assert calculate_semantic_density("   ") == 0.0
    
    def test_semantic_density_short_text(self):
        """Test semantic density with very short text."""
        density = calculate_semantic_density("Hi")
        assert density == 0.0  # Too short
    
    def test_semantic_density_repetitive_vs_diverse(self):
        """Test that diverse text has higher density than repetitive text."""
        repetitive = "the the the the the the the the"
        diverse = "machine learning algorithms neural networks artificial intelligence"
        
        density_rep = calculate_semantic_density(repetitive)
        density_div = calculate_semantic_density(diverse)
        
        assert density_div > density_rep


class TestChunkText:
    """Test text chunking functionality."""
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "This is a test document. " * 50  # Create longer text
        filename = "test.txt"
        
        chunks = chunk_text(text, filename, chunk_size=200, chunk_overlap=50)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 1  # Should create multiple chunks
        
        for chunk in chunks:
            assert isinstance(chunk, dict)
            assert 'content' in chunk
            assert 'metadata' in chunk
            assert chunk['metadata']['filename'] == filename
            assert len(chunk['content']) <= 250  # Allow some flexibility
    
    def test_chunk_text_empty(self):
        """Test chunking with empty text."""
        chunks = chunk_text("", "test.txt")
        assert chunks == []
    
    def test_chunk_text_short(self):
        """Test chunking with text shorter than chunk size."""
        text = "This is a short text."
        chunks = chunk_text(text, "test.txt", chunk_size=1000)
        
        assert len(chunks) == 1
        assert chunks[0]['content'] == text


class TestSplitLargeSentence:
    """Test large sentence splitting functionality."""
    
    def test_split_large_sentence_normal(self):
        """Test splitting with normal sized sentence."""
        sentence = "This is a normal sentence."
        result = split_large_sentence(sentence, max_size=100)
        
        assert result == [sentence]
    
    def test_split_large_sentence_oversized(self):
        """Test splitting with oversized sentence."""
        sentence = "word " * 50  # Create long sentence
        result = split_large_sentence(sentence, max_size=20)
        
        assert len(result) > 1
        assert all(len(chunk) <= 20 for chunk in result)
        assert "".join(result).replace(" ", "") == sentence.replace(" ", "")


class TestIsHeading:
    """Test heading detection functionality."""
    
    def test_is_heading_markdown(self):
        """Test detection of markdown headings."""
        assert is_heading("## This is a heading")
        assert is_heading("# Main heading")
        assert not is_heading("This is not a heading")
    
    def test_is_heading_numbered(self):
        """Test detection of numbered headings."""
        assert is_heading("1. Introduction")
        assert is_heading("2.1 Methodology")
        assert not is_heading("1 is a number")
    
    def test_is_heading_brackets(self):
        """Test detection of bracketed headings."""
        assert is_heading("[Chapter 1]")
        assert is_heading("[Section A]")
        assert not is_heading("This [is not] a heading")
    
    def test_is_heading_length_limits(self):
        """Test heading detection length limits."""
        assert not is_heading("Hi")  # Too short
        assert not is_heading("A" * 201)  # Too long
        assert is_heading("Valid Heading Length")


class TestIntegration:
    """Integration tests for text processing pipeline."""
    
    def test_full_processing_pipeline(self):
        """Test the complete text processing pipeline."""
        text = """
        # Introduction
        
        This is a sample document about machine learning. It contains multiple paragraphs
        and technical terms like neural networks, algorithms, and data processing.
        
        ## Methodology
        
        We use various techniques including supervised learning, unsupervised learning,
        and reinforcement learning approaches.
        """
        
        # Test the full pipeline
        chunks = chunk_text(text, "sample.txt", chunk_size=300, chunk_overlap=50)
        
        assert len(chunks) > 0
        for chunk in chunks:
            # Test that each chunk has proper metadata
            assert 'semantic_density' in chunk['metadata']
            assert 'keyword_tags' in chunk['metadata']
            assert 'contains_heading' in chunk['metadata']
            
            # Test keyword extraction
            keywords = chunk['metadata']['keyword_tags'].split(',') if chunk['metadata']['keyword_tags'] else []
            assert isinstance(keywords, list)


if __name__ == '__main__':
    pytest.main([__file__])
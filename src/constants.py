#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared Constants for RAG Document Analysis System.

This module contains shared constants, stopwords, and other reusable values
to avoid code duplication across the codebase.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

def get_config_value(config_path: str, default_value, config=None):
    """
    Helper function to get configuration values with fallback to defaults.
    
    Args:
        config_path: Dot-notation path to config value (e.g., 'chunking.max_file_size_mb')
        default_value: Default value if config not available
        config: Configuration object (will load if None)
    
    Returns:
        Configuration value or default
    """
    try:
        if config is None:
            from src.config import get_config
            config = get_config()
        
        # Navigate nested config path
        value = config
        for key in config_path.split('.'):
            if hasattr(value, key):
                value = getattr(value, key)
            else:
                return default_value
        return value
    except Exception:
        return default_value

# German and English stopwords for text processing
GERMAN_STOPWORDS = {
    'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich',
    'des', 'auf', 'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als',
    'auch', 'es', 'an', 'werden', 'aus', 'er', 'hat', 'dass', 'sie',
    'nach', 'wird', 'bei', 'einer', 'um', 'am', 'sind', 'noch', 'wie',
    'einem', 'über', 'einen', 'so', 'zum', 'war', 'haben', 'nur', 'oder',
    'aber', 'vor', 'zur', 'bis', 'mehr', 'durch', 'man', 'sein', 'wurde',
    'wenn', 'können', 'alle', 'würde', 'meine', 'macht', 'kann', 'soll',
    'wir', 'ich', 'dir', 'du', 'ihr', 'uns', 'euch'
}

ENGLISH_STOPWORDS = {
    'the', 'and', 'you', 'can', 'use', 'all', 'any', 'may', 'new', 'now', 'old',
    'see', 'two', 'way', 'who', 'its', 'did', 'get', 'has', 'had', 'let',
    'put', 'say', 'too', 'old', 'our', 'out', 'day', 'own', 'run', 'set',
    'this', 'that', 'with', 'for', 'are', 'was'
}

# Combined stopwords for multilingual text processing
COMBINED_STOPWORDS = GERMAN_STOPWORDS | ENGLISH_STOPWORDS

# File processing constants - configurable via config.yaml
def get_max_file_size_mb(config=None):
    return get_config_value('chunking.max_file_size_mb', 50, config)

def get_supported_file_formats(config=None):
    return get_config_value('chunking.supported_formats', ["txt", "pdf", "docx"], config)

# Text processing constants - configurable via config.yaml  
def get_min_keyword_length(config=None):
    return get_config_value('hybrid_retrieval.keyword_extraction.min_length', 3, config)

def get_max_keywords_default(config=None):
    return get_config_value('hybrid_retrieval.keyword_extraction.max_keywords', 5, config)

def get_min_chunk_length(config=None):
    return get_config_value('chunking.min_chunk_length', 100, config)

# Legacy constants for backward compatibility
DEFAULT_MAX_FILE_SIZE_MB = 50
SUPPORTED_FILE_FORMATS = ["txt", "pdf", "docx"]
MIN_KEYWORD_LENGTH = 3
MAX_KEYWORDS_DEFAULT = 5
MIN_CHUNK_LENGTH = 100

# Query analysis patterns
TECHNICAL_PATTERNS = [
    r'\b[A-Z_]{2,}\b',          # Constants/Env vars (API_KEY)
    r'\b\w+\(\)',               # Function calls
    r'\.\w+',                   # Attributes/Methods (.environ)
    r'\w+\.\w+',                # Module.function (os.environ)
    r'\bimport\s+\w+',          # Import statements
    r'\bfrom\s+\w+\s+import',   # From imports
    r'[\w-]+\.ya?ml\b',         # YAML files
    r'[\w-]+\.json\b',          # JSON files
    r'[\w-]+\.py\b',            # Python files
    r'[\w-]+\.js\b',            # JavaScript files
    r'\b\w+_\w+\b',             # Snake_case identifiers
]

CODE_KEYWORDS = {
    'python', 'kubernetes', 'docker', 'git', 'api', 'json', 
    'yaml', 'config', 'function', 'class', 'import', 'pip',
    'kubectl', 'pods', 'container', 'npm', 'node', 'java',
    'maven', 'gradle', 'bash', 'shell', 'script', 'command',
    'numpy', 'pandas', 'environment', 'variable', 'environ',
    'compose', 'dockerfile', 'befehl', 'datei'
}

QUESTION_PATTERNS = [
    r'\bwas\b', r'\bwie\b', r'\bwarum\b', r'\bwann\b', r'\?'
]

# Retrieval scoring constants - configurable via config.yaml
def get_default_semantic_weight(config=None):
    return get_config_value('retrieval.default_weights.semantic_weight', 0.6, config)

def get_default_keyword_weight(config=None):
    return get_config_value('retrieval.default_weights.keyword_weight', 0.4, config)

def get_hybrid_boost_factor(config=None):
    return get_config_value('hybrid_retrieval.hybrid_boost_factor', 1.2, config)

# BM25 parameters - configurable via config.yaml
def get_bm25_k1(config=None):
    return get_config_value('hybrid_retrieval.bm25.k1', 1.2, config)

def get_bm25_b(config=None):
    return get_config_value('hybrid_retrieval.bm25.b', 0.75, config)

def get_bm25_min_token_length(config=None):
    return get_config_value('hybrid_retrieval.bm25.min_token_length', 3, config)

# Legacy constants for backward compatibility
DEFAULT_SEMANTIC_WEIGHT = 0.6
DEFAULT_KEYWORD_WEIGHT = 0.4
HYBRID_BOOST_FACTOR = 1.2
BM25_K1 = 1.2
BM25_B = 0.75
BM25_MIN_TOKEN_LENGTH = 3

# Chunking strategy constants
CHUNK_STRATEGIES = {
    "simple": "Simple character-based chunking",
    "sentence_aware": "Sentence boundary-aware chunking", 
    "semantic": "Semantic boundary-aware chunking"
}

# UI Configuration constants
DEFAULT_UI_CONFIG = {
    "page_title": "RAG Document Analysis",
    "page_icon": "📚",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Logging format constants
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = "INFO"

# Performance constants
DEFAULT_BATCH_SIZE = 32
DEFAULT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_VECTOR_STORE_BATCH_SIZE = 1000
DEFAULT_CACHE_TTL_SECONDS = 3600

# Model defaults
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "google/flan-t5-base"
DEFAULT_EMBEDDING_DIMENSIONS = 384

if __name__ == '__main__':
    logger.info("Constants module loaded successfully")
    logger.info(f"Loaded {len(COMBINED_STOPWORDS)} stopwords")
    logger.info(f"Supported file formats: {SUPPORTED_FILE_FORMATS}")
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration Management System for RAG Document Analysis.

This module provides centralized configuration management with YAML-based
configuration files and environment variable override capabilities.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
import copy

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration settings."""
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    llm_model: str = "google/flan-t5-base"
    device_preference: str = "auto"
    trust_remote_code: bool = False
    use_auth_token: bool = False

@dataclass
class ChunkingConfig:
    """Text chunking configuration settings."""
    strategy: str = "sentence_aware"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size_mb: int = 50
    supported_formats: list = field(default_factory=lambda: ["txt", "pdf", "docx"])
    normalize_whitespace: bool = True
    remove_empty_lines: bool = True
    min_chunk_length: int = 100

@dataclass
class RetrievalConfig:
    """Retrieval configuration settings."""
    default_top_k: int = 5
    max_candidate_multiplier: int = 3
    similarity_threshold: float = 0.0
    default_weights: Dict[str, float] = field(default_factory=lambda: {
        "semantic_weight": 0.6,
        "keyword_weight": 0.4
    })

@dataclass
class HybridRetrievalConfig:
    """Enhanced hybrid retrieval configuration."""
    adaptive_weighting: bool = True
    query_type_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "technical": {"semantic_weight": 0.37, "keyword_weight": 0.63},
        "question": {"semantic_weight": 0.7, "keyword_weight": 0.3},
        "balanced": {"semantic_weight": 0.6, "keyword_weight": 0.4}
    })
    bm25: Dict[str, Any] = field(default_factory=lambda: {
        "k1": 1.2,
        "b": 0.75,
        "enable_stemming": True,
        "enable_compound_handling": True,
        "min_token_length": 3,
        "custom_stopwords": []
    })
    hybrid_boost_factor: float = 1.2
    keyword_extraction: Dict[str, Any] = field(default_factory=lambda: {
        "min_length": 3,
        "max_keywords": 5
    })

@dataclass
class GenerationConfig:
    """LLM generation configuration settings."""
    max_input_tokens: int = 2048
    max_output_tokens: int = 512
    temperature: float = 0.1
    do_sample: bool = False
    num_beams: int = 5
    early_stopping: bool = True
    system_prompt: str = "Du bist ein hilfsreicher Assistent, der Fragen basierend auf bereitgestellten Dokumenten beantwortet."
    answer_language: str = "german"
    no_context_response: str = "Entschuldigung, ich konnte keine relevanten Informationen zu Ihrer Frage finden."

@dataclass
class VectorDatabaseConfig:
    """Vector database configuration settings."""
    storage_path: str = "data/vectordb"
    default_collection: str = "documents"
    persist_directory: bool = True
    auto_persist: bool = True
    batch_size: int = 1000
    embedding_batch_size: int = 32

@dataclass
class FileProcessingConfig:
    """File processing configuration settings."""
    input_directory: str = "data/raw_texts"
    upload_directory: str = "data/uploads"
    create_directories: bool = True
    clean_temp_files: bool = True
    pdf: Dict[str, Any] = field(default_factory=lambda: {
        "primary_parser": "pdfplumber",
        "fallback_parser": "pymupdf",
        "extract_tables": True,
        "preserve_layout": True
    })
    docx: Dict[str, Any] = field(default_factory=lambda: {
        "preserve_formatting": True,
        "extract_tables": True,
        "include_headers": True
    })

@dataclass
class UIConfig:
    """Streamlit UI configuration settings."""
    page_title: str = "RAG Document Analysis"
    page_icon: str = "📚"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    default_retrieval_method: str = "hybrid"
    show_debug_info: bool = False
    enable_file_upload: bool = True
    max_context_chunks: int = 5
    expand_first_chunk: bool = True
    show_chunk_metadata: bool = True

@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = False
    log_file_path: str = "logs/rag_system.log"
    log_performance: bool = True
    log_queries: bool = False

@dataclass
class PerformanceConfig:
    """Performance and monitoring configuration settings."""
    enable_monitoring: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    embedding_timeout: int = 30
    llm_timeout: int = 60
    retrieval_timeout: int = 10
    clear_cache_on_restart: bool = True
    max_memory_usage_mb: int = 4096

@dataclass
class DevelopmentConfig:
    """Development and testing configuration settings."""
    debug_mode: bool = False
    enable_test_data: bool = False
    test_data_directory: str = "test_data"
    enable_benchmarking: bool = False
    benchmark_output_directory: str = "benchmarks/results"

class RAGConfig:
    """
    Centralized configuration management for the RAG system.
    
    Features:
    - YAML-based configuration files
    - Environment variable overrides
    - Type-safe configuration dataclasses
    - Validation and defaults
    - Runtime configuration updates
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file. Defaults to 'config.yaml'
        """
        self.config_path = config_path or self._find_config_file()
        self._raw_config = {}
        self._load_config()
        self._setup_logging()
    
    def _find_config_file(self) -> str:
        """Find the configuration file in standard locations."""
        possible_paths = [
            "config.yaml",
            "config.yml",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # If no config file found, create default
        default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        logger.warning(f"No config file found. Using defaults. Consider creating {default_path}")
        return default_path
    
    def _load_config(self):
        """Load configuration from YAML file with environment variable overrides."""
        # Load YAML configuration
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._raw_config = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading config file {self.config_path}: {e}")
                self._raw_config = {}
        else:
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            self._raw_config = {}
        
        # Apply environment variable overrides
        self._apply_environment_overrides()
        
        # Initialize configuration dataclasses
        self._initialize_configs()
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides to configuration."""
        env_prefix = "RAG_"
        
        for env_var, value in os.environ.items():
            if not env_var.startswith(env_prefix):
                continue
            
            # Parse environment variable name
            # Format: RAG_SECTION_KEY or RAG_SECTION_SUBSECTION_KEY
            config_path = env_var[len(env_prefix):].lower().split('_')
            
            if len(config_path) < 2:
                continue
            
            # Convert string value to appropriate type
            parsed_value = self._parse_env_value(value)
            
            # Apply to configuration
            self._set_nested_config(self._raw_config, config_path, parsed_value)
            
            logger.info(f"Applied environment override: {env_var} = {parsed_value}")
    
    def _parse_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Parse environment variable value to appropriate type."""
        # Boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer values
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float values
        try:
            return float(value)
        except ValueError:
            pass
        
        # String values
        return value
    
    def _set_nested_config(self, config: Dict, path: list, value: Any):
        """Set nested configuration value using dot notation path."""
        current = config
        
        # Navigate to parent of target key
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set final value
        current[path[-1]] = value
    
    def _initialize_configs(self):
        """Initialize configuration dataclasses from raw config."""
        self.models = self._create_config(ModelConfig, self._raw_config.get('models', {}))
        self.chunking = self._create_config(ChunkingConfig, self._raw_config.get('chunking', {}))
        self.retrieval = self._create_config(RetrievalConfig, self._raw_config.get('retrieval', {}))
        self.hybrid_retrieval = self._create_config(HybridRetrievalConfig, self._raw_config.get('hybrid_retrieval', {}))
        self.generation = self._create_config(GenerationConfig, self._raw_config.get('generation', {}))
        self.vector_database = self._create_config(VectorDatabaseConfig, self._raw_config.get('vector_database', {}))
        self.file_processing = self._create_config(FileProcessingConfig, self._raw_config.get('file_processing', {}))
        self.ui = self._create_config(UIConfig, self._raw_config.get('ui', {}))
        self.logging = self._create_config(LoggingConfig, self._raw_config.get('logging', {}))
        self.performance = self._create_config(PerformanceConfig, self._raw_config.get('performance', {}))
        self.development = self._create_config(DevelopmentConfig, self._raw_config.get('development', {}))
    
    def _create_config(self, config_class, config_data: Dict[str, Any]):
        """Create configuration dataclass instance with validation."""
        try:
            # Get default values
            default_instance = config_class()
            
            # Update with provided values
            for key, value in config_data.items():
                if hasattr(default_instance, key):
                    setattr(default_instance, key, value)
                else:
                    logger.warning(f"Unknown configuration key '{key}' for {config_class.__name__}")
            
            return default_instance
            
        except Exception as e:
            logger.error(f"Error creating {config_class.__name__}: {e}")
            return config_class()  # Return default instance
    
    def _setup_logging(self):
        """Setup logging based on configuration."""
        log_level = getattr(logging, self.logging.level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format=self.logging.format,
            handlers=[]
        )
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(self.logging.format)
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
        
        # Add file handler if configured
        if self.logging.log_to_file:
            os.makedirs(os.path.dirname(self.logging.log_file_path), exist_ok=True)
            file_handler = logging.FileHandler(self.logging.log_file_path)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
    
    def reload_config(self):
        """Reload configuration from file."""
        logger.info("Reloading configuration...")
        self._load_config()
    
    def get_section(self, section_name: str) -> Any:
        """Get configuration section by name."""
        return getattr(self, section_name, None)
    
    def update_section(self, section_name: str, updates: Dict[str, Any]):
        """Update configuration section at runtime."""
        section = self.get_section(section_name)
        if section is None:
            logger.error(f"Configuration section '{section_name}' not found")
            return
        
        for key, value in updates.items():
            if hasattr(section, key):
                setattr(section, key, value)
                logger.info(f"Updated {section_name}.{key} = {value}")
            else:
                logger.warning(f"Unknown configuration key '{key}' in section '{section_name}'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'models': self.models.__dict__,
            'chunking': self.chunking.__dict__,
            'retrieval': self.retrieval.__dict__,
            'hybrid_retrieval': self.hybrid_retrieval.__dict__,
            'generation': self.generation.__dict__,
            'vector_database': self.vector_database.__dict__,
            'file_processing': self.file_processing.__dict__,
            'ui': self.ui.__dict__,
            'logging': self.logging.__dict__,
            'performance': self.performance.__dict__,
            'development': self.development.__dict__
        }
    
    def save_config(self, output_path: Optional[str] = None):
        """Save current configuration to YAML file."""
        output_path = output_path or self.config_path
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {output_path}: {e}")
    
    def validate_config(self) -> bool:
        """Validate current configuration settings."""
        errors = []
        
        # Validate models
        if self.models.embedding_dimensions <= 0:
            errors.append("embedding_dimensions must be positive")
        
        # Validate chunking
        if self.chunking.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        if self.chunking.chunk_overlap >= self.chunking.chunk_size:
            errors.append("chunk_overlap must be less than chunk_size")
        
        # Validate retrieval
        if self.retrieval.default_top_k <= 0:
            errors.append("default_top_k must be positive")
        
        # Validate generation
        if self.generation.max_input_tokens <= 0:
            errors.append("max_input_tokens must be positive")
        if self.generation.max_output_tokens <= 0:
            errors.append("max_output_tokens must be positive")
        
        # Log errors
        for error in errors:
            logger.error(f"Configuration validation error: {error}")
        
        return len(errors) == 0

# Global configuration instance
_config_instance = None

def get_config() -> RAGConfig:
    """Get global configuration instance (singleton pattern)."""
    global _config_instance
    if _config_instance is None:
        _config_instance = RAGConfig()
    return _config_instance

def reload_config():
    """Reload global configuration."""
    global _config_instance
    if _config_instance is not None:
        _config_instance.reload_config()

def set_config_path(config_path: str):
    """Set configuration file path and reload."""
    global _config_instance
    _config_instance = RAGConfig(config_path)

# Example usage and testing
if __name__ == '__main__':
    # This section should be moved to a proper test file
    logger.info("Configuration system module loaded successfully")
    logger.info("Use this module by importing RAGConfig or get_config()")
    logger.info("For testing, run: python -m pytest tests/test_config.py")
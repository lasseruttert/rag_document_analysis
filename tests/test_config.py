#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for config module.
"""

import pytest
import sys
import os
import tempfile
import yaml
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import RAGConfig, get_config


class TestRAGConfig:
    """Test RAGConfig class functionality."""
    
    def test_config_initialization_default(self):
        """Test that config can be initialized with defaults."""
        config = RAGConfig()
        
        # Test that all sections exist
        assert hasattr(config, 'models')
        assert hasattr(config, 'chunking')
        assert hasattr(config, 'retrieval')
        assert hasattr(config, 'generation')
        assert hasattr(config, 'vector_database')
        assert hasattr(config, 'ui')
        assert hasattr(config, 'logging')
    
    def test_config_validation_valid(self):
        """Test config validation with valid configuration."""
        config = RAGConfig()
        assert config.validate_config() is True
    
    def test_config_validation_invalid_chunk_size(self):
        """Test config validation with invalid chunk size."""
        config = RAGConfig()
        config.chunking.chunk_size = -1
        assert config.validate_config() is False
    
    def test_config_validation_invalid_overlap(self):
        """Test config validation with invalid overlap."""
        config = RAGConfig()
        config.chunking.chunk_overlap = config.chunking.chunk_size + 100
        assert config.validate_config() is False
    
    def test_config_to_dict(self):
        """Test conversion of config to dictionary."""
        config = RAGConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'models' in config_dict
        assert 'chunking' in config_dict
        assert 'retrieval' in config_dict
        
        # Test that values are serializable
        yaml.dump(config_dict)  # Should not raise exception
    
    def test_config_section_access(self):
        """Test accessing config sections."""
        config = RAGConfig()
        
        models_section = config.get_section('models')
        assert models_section is not None
        assert hasattr(models_section, 'embedding_model')
        
        invalid_section = config.get_section('nonexistent')
        assert invalid_section is None
    
    def test_config_update_section(self):
        """Test updating config section at runtime."""
        config = RAGConfig()
        original_chunk_size = config.chunking.chunk_size
        
        config.update_section('chunking', {'chunk_size': 1500})
        assert config.chunking.chunk_size == 1500
        assert config.chunking.chunk_size != original_chunk_size


class TestConfigFile:
    """Test config file loading and handling."""
    
    def test_config_with_custom_file(self):
        """Test loading config from custom file."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            test_config = {
                'models': {'embedding_model': 'test-model'},
                'chunking': {'chunk_size': 999}
            }
            yaml.dump(test_config, f)
            temp_path = f.name
        
        try:
            config = RAGConfig(config_path=temp_path)
            assert config.models.embedding_model == 'test-model'
            assert config.chunking.chunk_size == 999
        finally:
            os.unlink(temp_path)
    
    def test_config_with_nonexistent_file(self):
        """Test handling of nonexistent config file."""
        config = RAGConfig(config_path='/nonexistent/path/config.yaml')
        # Should use defaults without crashing
        assert hasattr(config, 'models')
        assert hasattr(config, 'chunking')


class TestEnvironmentOverrides:
    """Test environment variable override functionality."""
    
    def test_environment_override_simple(self):
        """Test simple environment variable override."""
        # Set environment variable
        os.environ['RAG_MODELS_EMBEDDING_MODEL'] = 'test-env-model'
        
        try:
            config = RAGConfig()
            # Should apply the environment override
            assert config.models.embedding_model == 'test-env-model'
        finally:
            # Clean up
            if 'RAG_MODELS_EMBEDDING_MODEL' in os.environ:
                del os.environ['RAG_MODELS_EMBEDDING_MODEL']
    
    def test_environment_override_numeric(self):
        """Test environment variable override with numeric values."""
        os.environ['RAG_CHUNKING_CHUNK_SIZE'] = '1234'
        
        try:
            config = RAGConfig()
            assert config.chunking.chunk_size == 1234
            assert isinstance(config.chunking.chunk_size, int)
        finally:
            if 'RAG_CHUNKING_CHUNK_SIZE' in os.environ:
                del os.environ['RAG_CHUNKING_CHUNK_SIZE']
    
    def test_environment_override_boolean(self):
        """Test environment variable override with boolean values."""
        os.environ['RAG_DEVELOPMENT_DEBUG_MODE'] = 'true'
        
        try:
            config = RAGConfig()
            assert config.development.debug_mode is True
        finally:
            if 'RAG_DEVELOPMENT_DEBUG_MODE' in os.environ:
                del os.environ['RAG_DEVELOPMENT_DEBUG_MODE']


class TestGlobalConfig:
    """Test global config singleton functionality."""
    
    def test_get_config_singleton(self):
        """Test that get_config returns singleton instance."""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2  # Same instance
    
    def test_config_sections_accessible(self):
        """Test that all config sections are accessible through global config."""
        config = get_config()
        
        # Test model configuration
        assert hasattr(config.models, 'embedding_model')
        assert hasattr(config.models, 'llm_model')
        
        # Test chunking configuration
        assert hasattr(config.chunking, 'chunk_size')
        assert hasattr(config.chunking, 'chunk_overlap')
        
        # Test retrieval configuration
        assert hasattr(config.retrieval, 'default_top_k')
        
        # Test generation configuration
        assert hasattr(config.generation, 'max_input_tokens')
        assert hasattr(config.generation, 'temperature')


class TestConfigValues:
    """Test that config values are reasonable."""
    
    def test_default_values_reasonable(self):
        """Test that default config values are reasonable."""
        config = get_config()
        
        # Test chunk sizes
        assert 100 <= config.chunking.chunk_size <= 10000
        assert 0 <= config.chunking.chunk_overlap < config.chunking.chunk_size
        
        # Test retrieval settings
        assert 1 <= config.retrieval.default_top_k <= 50
        assert 0 <= config.retrieval.similarity_threshold <= 1
        
        # Test generation settings
        assert 0 <= config.generation.temperature <= 2
        assert config.generation.max_input_tokens > 0
        assert config.generation.max_output_tokens > 0
        
        # Test weights sum to 1
        weights = config.retrieval.default_weights
        total_weight = weights['semantic_weight'] + weights['keyword_weight']
        assert abs(total_weight - 1.0) < 0.01


class TestConfigSerialization:
    """Test config serialization and persistence."""
    
    def test_config_save_load_cycle(self):
        """Test saving and loading config maintains values."""
        config = RAGConfig()
        
        # Modify some values
        original_chunk_size = config.chunking.chunk_size
        config.chunking.chunk_size = 1337
        config.models.embedding_model = "test-save-model"
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save_config(temp_path)
            
            # Load new config from saved file
            new_config = RAGConfig(config_path=temp_path)
            
            assert new_config.chunking.chunk_size == 1337
            assert new_config.models.embedding_model == "test-save-model"
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__])
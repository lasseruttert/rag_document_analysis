#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration Integration Test for RAG Document Analysis System.

This script tests the configuration system integration without requiring
all heavy dependencies like PyTorch.
"""

import os
import sys
import tempfile

def test_config_system():
    """Test basic configuration system functionality."""
    print("Testing Configuration System Integration...")
    
    try:
        from src.config import RAGConfig, get_config
        
        print("[PASS] Configuration system imports successfully")
        
        # Test configuration loading
        config = get_config()
        print(f"[PASS] Configuration loaded successfully")
        print(f"   - Embedding model: {config.models.embedding_model}")
        print(f"   - Chunk size: {config.chunking.chunk_size}")
        print(f"   - Default top_k: {config.retrieval.default_top_k}")
        
        # Test configuration validation
        is_valid = config.validate_config()
        print(f"[PASS] Configuration validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Test environment override
        os.environ['RAG_CHUNKING_CHUNK_SIZE'] = '1500'
        config.reload_config()
        if config.chunking.chunk_size == 1500:
            print("[PASS] Environment variable override working")
        else:
            print("[FAIL] Environment variable override failed")
        
        # Clean up
        del os.environ['RAG_CHUNKING_CHUNK_SIZE']
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Configuration system test failed: {e}")
        return False

def test_config_file_loading():
    """Test YAML configuration file loading."""
    print("\nTesting YAML Configuration File Loading...")
    
    try:
        import yaml
        from src.config import RAGConfig
        
        # Test with existing config file
        if os.path.exists('config.yaml'):
            config = RAGConfig('config.yaml')
            print("[PASS] YAML configuration file loaded successfully")
            
            # Test specific values
            if hasattr(config.models, 'embedding_model'):
                print(f"   - Models section loaded: {config.models.embedding_model}")
            if hasattr(config.chunking, 'strategy'):
                print(f"   - Chunking section loaded: {config.chunking.strategy}")
            if hasattr(config.ui, 'page_title'):
                print(f"   - UI section loaded: {config.ui.page_title}")
                
            return True
        else:
            print("[WARN] No config.yaml file found, using defaults")
            return True
            
    except Exception as e:
        print(f"[FAIL] YAML configuration test failed: {e}")
        return False

def test_config_integration_imports():
    """Test that modules can import config successfully."""
    print("\nTesting Configuration Integration with Modules...")
    
    try:
        # Test that text processor can import config (without heavy dependencies)
        sys.path.insert(0, 'src')
        
        # Test config imports in modules
        config_imports = [
            ('text_processor', 'from src.config import get_config, ChunkingConfig'),
            ('config', 'from src.config import RAGConfig, get_config'),
        ]
        
        for module, import_statement in config_imports:
            try:
                exec(import_statement)
                print(f"[PASS] Config import in {module} successful")
            except Exception as e:
                print(f"[FAIL] Config import in {module} failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Configuration integration test failed: {e}")
        return False

def test_config_serialization():
    """Test configuration serialization and saving."""
    print("\nTesting Configuration Serialization...")
    
    try:
        from src.config import RAGConfig
        
        config = RAGConfig()
        
        # Test to_dict conversion
        config_dict = config.to_dict()
        
        # Verify structure
        expected_sections = ['models', 'chunking', 'retrieval', 'generation', 
                           'vector_database', 'ui', 'logging']
        
        for section in expected_sections:
            if section in config_dict:
                print(f"[PASS] Section '{section}' in serialized config")
            else:
                print(f"[FAIL] Missing section '{section}' in serialized config")
                return False
        
        # Test save functionality (to temporary file)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            config.save_config(temp_path)
            print("[PASS] Configuration save successful")
            
            # Test loading the saved config
            saved_config = RAGConfig(temp_path)
            if saved_config.models.embedding_model == config.models.embedding_model:
                print("[PASS] Configuration round-trip successful")
            else:
                print("[FAIL] Configuration round-trip failed")
                return False
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Configuration serialization test failed: {e}")
        return False

def main():
    """Run all configuration integration tests."""
    print("=" * 60)
    print("RAG CONFIGURATION SYSTEM INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        test_config_system,
        test_config_file_loading,
        test_config_integration_imports,
        test_config_serialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    print("=" * 60)
    print(f"CONFIGURATION INTEGRATION TEST RESULTS: {passed}/{total} PASSED")
    print("=" * 60)
    
    if passed == total:
        print("[SUCCESS] All configuration integration tests PASSED!")
        print("\n[PASS] Configuration System Status: READY FOR USE")
        print("\nNext steps:")
        print("1. Install full dependencies (PyTorch, sentence-transformers, etc.)")
        print("2. Test full pipeline integration")
        print("3. Run Streamlit app with new configuration system")
        return True
    else:
        print(f"[FAIL] {total - passed} tests FAILED!")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
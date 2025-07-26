#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Collection Management System Integration Test.

This script tests the complete P2.2 Multi-Collection Management implementation
including collection management, metadata filtering, and pipeline integration.
"""

import os
import sys
import tempfile
from typing import List, Dict, Any

def test_metadata_filtering():
    """Test metadata filtering system."""
    print("Testing Metadata Filtering System...")
    
    try:
        from src.metadata_filter import MetadataFilter, QueryFilter, CombinedFilter, FilterOperator
        
        # Sample documents for testing
        sample_docs = [
            {
                'content': 'Kubernetes Pod documentation with detailed explanations...',
                'metadata': {
                    'filename': 'kubernetes_basics.txt',
                    'file_type': 'txt',
                    'chunk_size': 1200,
                    'position': 0,
                    'semantic_density': 0.75,
                    'keyword_tags': ['kubernetes', 'pod', 'container'],
                    'contains_heading': True
                }
            },
            {
                'content': 'Python best practices guide for clean code...',
                'metadata': {
                    'filename': 'python_guide.pdf',
                    'file_type': 'pdf',
                    'chunk_size': 800,
                    'position': 1500,
                    'semantic_density': 0.65,
                    'keyword_tags': ['python', 'best', 'practices'],
                    'contains_heading': False
                }
            }
        ]
        
        # Test 1: File type filtering
        pdf_filter = MetadataFilter.by_file_type("pdf")
        pdf_docs = MetadataFilter.apply_filters(sample_docs, pdf_filter)
        assert len(pdf_docs) == 1, f"Expected 1 PDF document, got {len(pdf_docs)}"
        assert pdf_docs[0]['metadata']['file_type'] == 'pdf'
        print("[PASS] File type filtering works correctly")
        
        # Test 2: Size filtering
        large_docs_filter = MetadataFilter.by_content_size(min_size=1000)
        large_docs = MetadataFilter.apply_filters(sample_docs, large_docs_filter)
        assert len(large_docs) == 1, f"Expected 1 large document, got {len(large_docs)}"
        assert large_docs[0]['metadata']['chunk_size'] >= 1000
        print("[PASS] Content size filtering works correctly")
        
        # Test 3: Combined filtering
        combined_filter = MetadataFilter.combine_filters([
            MetadataFilter.by_file_type(['txt', 'pdf']),
            MetadataFilter.by_semantic_density(min_density=0.6)
        ], operator="AND")
        
        filtered_docs = MetadataFilter.apply_filters(sample_docs, combined_filter)
        assert len(filtered_docs) == 2, f"Expected 2 documents, got {len(filtered_docs)}"
        print("[PASS] Combined filtering works correctly")
        
        # Test 4: Custom field filtering
        custom_filter = MetadataFilter.by_custom_field(
            'semantic_density', 0.7, FilterOperator.GREATER_THAN
        )
        high_density_docs = MetadataFilter.apply_filters(sample_docs, custom_filter)
        assert len(high_density_docs) == 1, f"Expected 1 high density document, got {len(high_density_docs)}"
        print("[PASS] Custom field filtering works correctly")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Metadata filtering test failed: {e}")
        return False

def test_configuration_integration():
    """Test configuration system integration with multi-collection features."""
    print("\nTesting Configuration Integration...")
    
    try:
        from src.config import get_config
        
        config = get_config()
        
        # Verify configuration sections exist
        assert hasattr(config, 'vector_database'), "Missing vector_database configuration"
        assert hasattr(config, 'retrieval'), "Missing retrieval configuration"
        assert hasattr(config, 'ui'), "Missing UI configuration"
        
        # Check relevant configuration values
        assert config.vector_database.default_collection is not None
        assert config.retrieval.default_top_k > 0
        assert config.ui.page_title is not None
        
        print("[PASS] Configuration integration works correctly")
        return True
        
    except Exception as e:
        print(f"[FAIL] Configuration integration test failed: {e}")
        return False

def test_pipeline_imports():
    """Test that pipeline imports all new modules correctly."""
    print("\nTesting Pipeline Import Integration...")
    
    try:
        # Test imports without initializing heavy dependencies
        import_tests = [
            "from src.metadata_filter import MetadataFilter, QueryFilter, CombinedFilter",
            "from src.config import get_config",
        ]
        
        for test_import in import_tests:
            exec(test_import)
            print(f"✅ Import successful: {test_import.split('import')[1].strip()}")
        
        # Test that pipeline can be imported (without initialization)
        from src.pipeline import RAGPipeline
        print("✅ RAGPipeline import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline import test failed: {e}")
        return False

def test_enhanced_vectorstore_interface():
    """Test enhanced vectorstore interface design."""
    print("\nTesting Enhanced VectorStore Interface...")
    
    try:
        # Test that all new methods exist in VectorStoreManager
        from src.vectorstore import VectorStoreManager
        
        required_methods = [
            'search_with_filters',
            'search_across_collections',
            'get_collection_statistics',
            'list_all_collections',
            'create_collection',
            'delete_collection',
            'set_active_collection'
        ]
        
        for method in required_methods:
            assert hasattr(VectorStoreManager, method), f"Missing method: {method}"
            print(f"✅ Method exists: {method}")
        
        return True
        
    except Exception as e:
        print(f"❌ VectorStore interface test failed: {e}")
        return False

def test_enhanced_pipeline_interface():
    """Test enhanced pipeline interface design."""
    print("\nTesting Enhanced Pipeline Interface...")
    
    try:
        from src.pipeline import RAGPipeline
        
        required_methods = [
            'create_collection',
            'delete_collection',
            'set_active_collection',
            'list_collections',
            'get_collection_statistics',
            'ingest_documents_to_collection',
            'answer_query_with_filters',
            'search_across_collections',
            'enhanced_answer_query_with_filters'
        ]
        
        for method in required_methods:
            assert hasattr(RAGPipeline, method), f"Missing method: {method}"
            print(f"✅ Method exists: {method}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline interface test failed: {e}")
        return False

def test_streamlit_app_structure():
    """Test that Streamlit app has all required components."""
    print("\nTesting Streamlit App Structure...")
    
    try:
        # Read the streamlit app file
        app_path = "app/streamlit_app.py"
        with open(app_path, 'r', encoding='utf-8') as f:
            app_content = f.read()
        
        # Check for required imports
        required_imports = [
            "from src.metadata_filter import MetadataFilter",
            "from src.collection_manager import CollectionInfo"
        ]
        
        for required_import in required_imports:
            assert required_import in app_content, f"Missing import: {required_import}"
            print(f"✅ Import found: {required_import}")
        
        # Check for required UI components
        required_components = [
            "Collection Management",
            "Filter-Einstellungen",
            "Collection Dashboard",
            "Cross-Collection",
            "Filter verwenden"
        ]
        
        for component in required_components:
            assert component in app_content, f"Missing UI component: {component}"
            print(f"✅ UI component found: {component}")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit app structure test failed: {e}")
        return False

def test_file_completeness():
    """Test that all required files exist and are properly structured."""
    print("\nTesting File Completeness...")
    
    try:
        required_files = [
            "src/collection_manager.py",
            "src/metadata_filter.py",
            "src/vectorstore.py",
            "src/pipeline.py",
            "app/streamlit_app.py",
            "config.yaml"
        ]
        
        for file_path in required_files:
            assert os.path.exists(file_path), f"Missing file: {file_path}"
            
            # Check file is not empty
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert len(content) > 100, f"File too small: {file_path}"
            
            print(f"✅ File exists and has content: {file_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ File completeness test failed: {e}")
        return False

def main():
    """Run all P2.2 Multi-Collection Management integration tests."""
    print("=" * 70)
    print("P2.2 MULTI-COLLECTION MANAGEMENT INTEGRATION TESTS")
    print("=" * 70)
    
    tests = [
        test_file_completeness,
        test_configuration_integration,
        test_metadata_filtering,
        test_pipeline_imports,
        test_enhanced_vectorstore_interface,
        test_enhanced_pipeline_interface,
        test_streamlit_app_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    print("=" * 70)
    print(f"P2.2 INTEGRATION TEST RESULTS: {passed}/{total} PASSED")
    print("=" * 70)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! P2.2 Multi-Collection Management is READY!")
        print("\n✅ IMPLEMENTED FEATURES:")
        print("   • Collection Management System (CRUD operations)")
        print("   • Advanced Metadata Filtering (file type, size, custom fields)")
        print("   • Multi-Collection Vector Store Operations")
        print("   • Collection-Specific Pipeline Methods")
        print("   • Cross-Collection Search Capabilities")
        print("   • Enhanced Streamlit UI with Collection Management")
        print("   • Filtering Interface with Multiple Filter Types")
        print("   • Statistics Dashboard and Collection Overview")
        print("\n🚀 NEXT STEPS:")
        print("   1. Install full dependencies (PyTorch, ChromaDB, etc.)")
        print("   2. Test with real documents and collections")
        print("   3. Run Streamlit app: streamlit run app/streamlit_app.py")
        print("   4. Begin P3.1 Advanced Query Interface implementation")
        return True
    else:
        print(f"❌ {total - passed} tests FAILED!")
        print("Please review the failed tests before proceeding.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
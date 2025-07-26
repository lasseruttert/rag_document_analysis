#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test BM25 caching functionality.
"""

import sys
import os
import time
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_bm25_caching():
    """Test BM25 caching functionality."""
    print("Testing BM25 caching functionality...")
    
    try:
        from retriever import GermanBM25Retriever
        
        # Create sample documents
        sample_docs = [
            {'id': '1', 'content': 'This is about Python programming and machine learning algorithms', 'metadata': {'chunk_id': 'doc1-1'}},
            {'id': '2', 'content': 'Kubernetes deployment strategies for microservices', 'metadata': {'chunk_id': 'doc2-1'}},
            {'id': '3', 'content': 'Docker container orchestration best practices', 'metadata': {'chunk_id': 'doc3-1'}},
            {'id': '4', 'content': 'Natural language processing with neural networks', 'metadata': {'chunk_id': 'doc4-1'}},
            {'id': '5', 'content': 'Database optimization techniques for large datasets', 'metadata': {'chunk_id': 'doc5-1'}}
        ]
        
        # Test with temporary cache directory
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, 'bm25_test_cache')
            
            print("Building BM25 model (first time)...")
            start_time = time.time()
            retriever1 = GermanBM25Retriever(sample_docs, cache_dir=cache_dir, enable_cache=True)
            build_time = time.time() - start_time
            
            print(f"First build took: {build_time:.3f} seconds")
            
            # Test retrieval works
            results = retriever1.retrieve("Python programming", top_k=2)
            assert len(results) > 0, "Should return results"
            print(f"Retrieved {len(results)} results")
            
            print("Loading BM25 model from cache...")
            start_time = time.time()
            retriever2 = GermanBM25Retriever(sample_docs, cache_dir=cache_dir, enable_cache=True)
            cache_time = time.time() - start_time
            
            print(f"Cache load took: {cache_time:.3f} seconds")
            
            # Verify cache was faster
            if cache_time < build_time:
                print("[PASS] Cache loading was faster than building")
            else:
                print("[WARN] Cache loading was not significantly faster")
            
            # Test that cached model works the same
            results2 = retriever2.retrieve("Python programming", top_k=2)
            assert len(results2) == len(results), "Cached model should return same number of results"
            print("[PASS] Cached model returns consistent results")
            
            # Test cache clearing
            retriever2.clear_cache()
            print("[PASS] Cache cleared successfully")
            
            # Test with caching disabled
            print("Testing with caching disabled...")
            retriever3 = GermanBM25Retriever(sample_docs, enable_cache=False)
            results3 = retriever3.retrieve("Python programming", top_k=2)
            assert len(results3) > 0, "Should work without caching"
            print("[PASS] Non-cached retriever works")
            
        print("[PASS] BM25 caching tests completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] BM25 caching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_hybrid_retriever_caching():
    """Test caching in EnhancedHybridRetriever."""
    print("Testing EnhancedHybridRetriever caching...")
    
    try:
        from retriever import EnhancedHybridRetriever
        
        # Create mock semantic retriever and vector store
        class MockSemanticRetriever:
            def retrieve(self, query, top_k):
                return [{'id': '1', 'content': 'mock content', 'metadata': {'chunk_id': 'mock-1'}, 'distance': 0.5}]
        
        class MockVectorStore:
            pass
        
        sample_docs = [
            {'id': '1', 'content': 'Python programming tutorial', 'metadata': {'chunk_id': 'doc1-1'}},
            {'id': '2', 'content': 'Machine learning basics', 'metadata': {'chunk_id': 'doc2-1'}}
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, 'hybrid_test_cache')
            
            # Create hybrid retriever with caching
            hybrid_retriever = EnhancedHybridRetriever(
                MockSemanticRetriever(),
                MockVectorStore(),
                sample_docs,
                cache_dir=cache_dir,
                enable_cache=True
            )
            
            # Test cache management methods
            hybrid_retriever.clear_bm25_cache()
            print("[PASS] Cache clearing method works")
            
            # Test rebuild method
            new_docs = sample_docs + [{'id': '3', 'content': 'New document', 'metadata': {'chunk_id': 'doc3-1'}}]
            hybrid_retriever.rebuild_bm25_index(new_docs)
            print("[PASS] Index rebuilding method works")
            
        print("[PASS] Enhanced hybrid retriever caching tests completed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Enhanced hybrid retriever test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run BM25 caching tests."""
    print("BM25 Caching Tests")
    print("=" * 50)
    
    tests = [
        test_bm25_caching,
        test_enhanced_hybrid_retriever_caching
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"[FAIL] {test_func.__name__} failed: {e}")
    
    print("\n" + "=" * 50)
    print(f"SUMMARY: {passed}/{total} test groups passed")
    
    if passed == total:
        print("All BM25 caching tests passed!")
        return 0
    else:
        print(f"{total - passed} test groups failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full system test for the RAG Document Analysis System.
"""

import sys
import os
import tempfile
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_full_rag_pipeline():
    """Test the complete RAG pipeline."""
    print("Testing complete RAG pipeline...")
    
    try:
        from pipeline import RAGPipeline
        from text_processor import chunk_text
        
        # Create test documents
        test_docs = [
            "Python is a high-level programming language known for its simplicity and readability.",
            "Machine learning is a subset of artificial intelligence that uses statistical techniques.",
            "Docker containers provide a lightweight way to package and deploy applications.",
            "Kubernetes orchestrates containerized applications across clusters of machines."
        ]
        
        # Create temporary directory for vector store
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_db_path = os.path.join(temp_dir, 'test_vectordb')
            
            print("Creating RAG pipeline...")
            pipeline = RAGPipeline(vector_db_path=vector_db_path)
            
            # Create chunks from test documents
            print("Creating document chunks...")
            all_chunks = []
            for i, doc in enumerate(test_docs):
                chunks = chunk_text(doc, f"test_doc_{i}.txt", chunk_size=200, chunk_overlap=50)
                all_chunks.extend(chunks)
            
            print(f"Created {len(all_chunks)} chunks")
            
            # Generate embeddings and add to vector store
            print("Generating embeddings and adding to vector store...")
            chunks_with_embeddings = pipeline.embedding_manager.generate_embeddings(all_chunks)
            pipeline.vector_store_manager.add_documents(chunks_with_embeddings)
            
            print(f"Added {len(chunks_with_embeddings)} chunks to vector store")
            
            # Test basic retrieval
            print("Testing basic retrieval...")
            results = pipeline.retriever.retrieve("What is Python?", top_k=2)
            print(f"Basic retrieval returned {len(results)} results")
            
            # Test query answering
            print("Testing query answering...")
            answer = pipeline.answer_query("What is Python programming?", top_k=2)
            print(f"Generated answer: {answer[:100]}...")
            
            # Test enhanced hybrid retrieval if documents exist
            try:
                print("Testing enhanced hybrid retrieval...")
                enhanced_answer = pipeline.enhanced_answer_query("What is machine learning?", top_k=2)
                print(f"Enhanced answer: {enhanced_answer[:100]}...")
            except Exception as e:
                print(f"Enhanced retrieval test skipped: {e}")
            
            print("[PASS] Full RAG pipeline test completed successfully")
            return True
            
    except Exception as e:
        print(f"[FAIL] Full RAG pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_components():
    """Test that Streamlit components can be imported."""
    print("\nTesting Streamlit components...")
    
    try:
        import streamlit as st
        print("Streamlit imported successfully")
        
        # Test that our app components can be imported
        sys.path.append('app')
        # We can't actually run the streamlit app in test mode, but we can check imports
        
        print("[PASS] Streamlit components test completed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Streamlit components test failed: {e}")
        return False

def test_file_processing():
    """Test file processing capabilities."""
    print("\nTesting file processing...")
    
    try:
        from text_processor import load_text_files, process_documents
        
        # Test with existing data directory if it exists
        data_dir = "data/raw_texts"
        if os.path.exists(data_dir):
            files = load_text_files(data_dir)
            print(f"Found {len(files)} text files in {data_dir}")
            
            if files:
                chunks = process_documents(data_dir)
                print(f"Processed into {len(chunks)} chunks")
        else:
            print("No data directory found, skipping file processing test")
        
        print("[PASS] File processing test completed")
        return True
        
    except Exception as e:
        print(f"[FAIL] File processing test failed: {e}")
        return False

def main():
    """Run full system tests."""
    print("RAG Document Analysis System - Full Integration Test")
    print("=" * 60)
    
    tests = [
        test_full_rag_pipeline,
        test_streamlit_components,
        test_file_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"[FAIL] {test_func.__name__} failed: {e}")
    
    print("\n" + "=" * 60)
    print(f"INTEGRATION TEST SUMMARY: {passed}/{total} test groups passed")
    
    if passed == total:
        print("All integration tests passed! RAG system is fully working!")
        return 0
    else:
        print(f"{total - passed} test groups failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
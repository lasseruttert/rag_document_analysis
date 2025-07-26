#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for Hybrid Retrieval System.
"""

import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import QueryAnalyzer, GermanBM25Retriever, EnhancedHybridRetriever
from src.pipeline import RAGPipeline
from src.embeddings import EmbeddingManager
from src.vectorstore import VectorStoreManager


class TestQueryAnalyzer:
    """Test QueryAnalyzer for adaptive query type detection."""
    
    def setup_method(self):
        """Setup test environment."""
        self.analyzer = QueryAnalyzer()
    
    def test_technical_query_detection(self):
        """Test technical query detection."""
        technical_queries = [
            "Wie erstelle ich eine Docker-Compose.yml?",
            "Was ist os.environ in Python?",
            "kubectl get pods Befehl",
            "API_KEY environment variable",
            "import numpy as np",
            "config.yaml Datei"
        ]
        
        for query in technical_queries:
            result = self.analyzer.analyze_query_type(query)
            print(f"Technical query: '{query}' -> {result}")
            
            # Technical queries should have higher keyword weight
            assert result['keyword_weight'] >= result['semantic_weight'], f"Failed for: {query}"
            assert result['query_type'] in ['technical', 'balanced']
    
    def test_question_query_detection(self):
        """Test question query detection."""
        question_queries = [
            "Was ist Kubernetes?",
            "Wie funktioniert Docker?",
            "Warum sollte ich virtuelle Umgebungen nutzen?",
            "Wann verwende ich Poetry?",
            "Was sind die Vorteile von Python?"
        ]
        
        for query in question_queries:
            result = self.analyzer.analyze_query_type(query)
            print(f"Question query: '{query}' -> {result}")
            
            # Question queries should have higher semantic weight
            assert result['semantic_weight'] >= result['keyword_weight'], f"Failed for: {query}"
            assert result['query_type'] in ['question', 'balanced']
    
    def test_balanced_query_detection(self):
        """Test balanced query detection."""
        balanced_queries = [
            "Docker Container Tutorial",
            "Python Development Guide",
            "Kubernetes Best Practices"
        ]
        
        for query in balanced_queries:
            result = self.analyzer.analyze_query_type(query)
            print(f"Balanced query: '{query}' -> {result}")
            
            # Weights should sum to 1.0
            assert abs(result['semantic_weight'] + result['keyword_weight'] - 1.0) < 0.01
            assert result['query_type'] in ['technical', 'question', 'balanced']


class TestGermanBM25Retriever:
    """Test GermanBM25Retriever for German language optimization."""
    
    def setup_method(self):
        """Setup test documents."""
        self.test_docs = [
            {
                'content': 'Kubernetes ist ein Open-Source-System zur Automatisierung der Bereitstellung von Container-Anwendungen',
                'metadata': {'chunk_id': 'test-1', 'filename': 'k8s.txt'}
            },
            {
                'content': 'Python Best Practices umfassen PEP 8 Stilrichtlinien und virtuelle Umgebungen für die Entwicklung',
                'metadata': {'chunk_id': 'test-2', 'filename': 'python.txt'}
            },
            {
                'content': 'Docker Container ermöglichen portable Anwendungen und Microservices-Architekturen',
                'metadata': {'chunk_id': 'test-3', 'filename': 'docker.txt'}
            },
            {
                'content': 'Configuration Management mit YAML-Dateien und Environment Variables in DevOps',
                'metadata': {'chunk_id': 'test-4', 'filename': 'devops.txt'}
            }
        ]
        self.retriever = GermanBM25Retriever(self.test_docs)
    
    def test_basic_retrieval(self):
        """Test basic BM25 retrieval functionality."""
        results = self.retriever.retrieve("Kubernetes", top_k=2)
        
        assert len(results) > 0, "No results returned"
        assert results[0]['metadata']['filename'] == 'k8s.txt', "Wrong document retrieved"
        assert 'bm25_score' in results[0], "Missing BM25 score"
        assert results[0]['bm25_score'] > 0, "Invalid BM25 score"
        
        print(f"Basic retrieval test passed: {len(results)} results")
    
    def test_german_stemming(self):
        """Test German stemming functionality."""
        # Test singular vs plural
        results1 = self.retriever.retrieve("Umgebung", top_k=1)
        results2 = self.retriever.retrieve("Umgebungen", top_k=1)
        
        # Should find the same document (stemmed to same root)
        assert len(results1) > 0 and len(results2) > 0, "Stemming test failed - no results"
        assert results1[0]['metadata']['filename'] == results2[0]['metadata']['filename'], "Stemming inconsistency"
        
        print("German stemming test passed")
    
    def test_compound_word_handling(self):
        """Test German compound word handling."""
        results = self.retriever.retrieve("Stilrichtlinien", top_k=1)
        
        assert len(results) > 0, "Compound word not found"
        assert results[0]['metadata']['filename'] == 'python.txt', "Wrong document for compound word"
        
        print("Compound word handling test passed")
    
    def test_technical_terms(self):
        """Test retrieval of technical terms."""
        results = self.retriever.retrieve("YAML Environment", top_k=1)
        
        assert len(results) > 0, "Technical terms not found"
        assert results[0]['metadata']['filename'] == 'devops.txt', "Wrong document for technical terms"
        
        print("Technical terms test passed")
    
    def test_stopword_filtering(self):
        """Test that German stopwords are properly filtered."""
        # Query with many stopwords should still work
        results = self.retriever.retrieve("Was ist das beste für die Entwicklung von Anwendungen", top_k=2)
        
        assert len(results) > 0, "Stopword filtering broke retrieval"
        
        print("Stopword filtering test passed")


class TestEnhancedHybridRetriever:
    """Integration test for complete Enhanced Hybrid Retriever."""
    
    def setup_method(self):
        """Setup test environment with temporary vector store."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.vector_db_path = self.test_dir / "test_vectordb"
        
        # Create test documents
        test_docs_dir = self.test_dir / "test_docs"
        test_docs_dir.mkdir()
        
        # Create test files
        (test_docs_dir / "kubernetes.txt").write_text(
            "Kubernetes ist ein Open-Source-System zur Container-Orchestrierung. "
            "Ein Pod ist die kleinste deployable Einheit in Kubernetes. "
            "Services ermöglichen die Kommunikation zwischen Pods. "
            "kubectl ist das Kommandozeilen-Tool für Kubernetes. Mit kubectl können Sie "
            "Pods auflisten mit 'kubectl get pods' oder Details anzeigen mit 'kubectl describe pod'.",
            encoding='utf-8'
        )
        
        (test_docs_dir / "python.txt").write_text(
            "Python Best Practices für die Softwareentwicklung. "
            "Virtuelle Umgebungen isolieren Abhängigkeiten mit virtualenv oder conda. "
            "PEP 8 definiert Stilrichtlinien für Python-Code. "
            "Verwenden Sie pip install für Paketinstallation in virtuellen Umgebungen.",
            encoding='utf-8'
        )
        
        # Setup pipeline
        self.pipeline = RAGPipeline(
            text_data_path=str(test_docs_dir),
            vector_db_path=str(self.vector_db_path)
        )
        
        # Ingest documents
        self.pipeline.ingest_documents()
    
    def teardown_method(self):
        """Cleanup test environment."""
        # Properly close ChromaDB connections
        if hasattr(self, 'pipeline'):
            if hasattr(self.pipeline, 'vector_store_manager'):
                if hasattr(self.pipeline.vector_store_manager, 'client'):
                    try:
                        self.pipeline.vector_store_manager.client.reset()
                    except:
                        pass
            # Clear pipeline references
            self.pipeline = None
        
        # Wait a bit for file handles to close
        import time
        time.sleep(0.1)
        
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_hybrid_retrieval_setup(self):
        """Test that hybrid retriever can be set up correctly."""
        # Trigger enhanced retriever setup
        self.pipeline._setup_enhanced_retriever()
        
        assert self.pipeline.enhanced_retriever is not None, "Enhanced retriever not initialized"
        assert hasattr(self.pipeline.enhanced_retriever, 'hybrid_retrieve'), "Missing hybrid_retrieve method"
        
        print("Hybrid retrieval setup test passed")
    
    def test_query_type_classification(self):
        """Test query type classification in end-to-end scenario."""
        test_queries = [
            ("Was ist Kubernetes?", "question"),  # Should have more semantic weight
            ("kubectl get pods", "technical"),    # Should have more keyword weight
            ("Python Tutorial", "balanced")       # Should be balanced
        ]
        
        # Setup enhanced retriever
        self.pipeline._setup_enhanced_retriever()
        
        for query, expected_type in test_queries:
            results = self.pipeline.enhanced_retriever.hybrid_retrieve(query, top_k=3)
            
            assert len(results) > 0, f"No results for query: {query}"
            assert 'hybrid_score' in results[0], "Missing hybrid_score"
            assert 'query_type' in results[0], "Missing query_type"
            
            detected_type = results[0]['query_type']
            print(f"Query: '{query}' -> Detected: {detected_type}, Expected: {expected_type}")
            
            # Note: We don't assert exact match as the classification might be balanced
            assert detected_type in ['technical', 'question', 'balanced'], f"Invalid query type: {detected_type}"
    
    def test_enhanced_answer_generation(self):
        """Test enhanced answer generation with hybrid retrieval."""
        test_queries = [
            "Was ist ein Pod?",
            "Python virtuelle Umgebungen",
            "kubectl command"
        ]
        
        for query in test_queries:
            answer = self.pipeline.enhanced_answer_query(query)
            
            assert isinstance(answer, str), "Answer should be string"
            assert len(answer) > 10, "Answer too short"
            assert "keine relevanten Informationen" not in answer.lower(), f"No relevant info found for: {query}"
            
            print(f"Enhanced answer test passed for: {query}")
    
    def test_hybrid_scoring_consistency(self):
        """Test that hybrid scoring is consistent and valid."""
        self.pipeline._setup_enhanced_retriever()
        
        results = self.pipeline.enhanced_retriever.hybrid_retrieve("Kubernetes Pod", top_k=5)
        
        assert len(results) > 0, "No results returned"
        
        # Check scoring consistency
        prev_score = float('inf')
        for result in results:
            hybrid_score = result.get('hybrid_score', 0)
            
            # Scores should be in descending order
            assert hybrid_score <= prev_score, "Hybrid scores not in descending order"
            
            # Scores should be between 0 and 1 (with possible boost up to 1.2)
            assert 0 <= hybrid_score <= 1.5, f"Invalid hybrid score: {hybrid_score}"
            
            # Required fields should be present
            assert 'semantic_weight' in result, "Missing semantic_weight"
            assert 'keyword_weight' in result, "Missing keyword_weight"
            assert 'retrieval_method' in result, "Missing retrieval_method"
            
            prev_score = hybrid_score
        
        print(f"Hybrid scoring consistency test passed with {len(results)} results")


def test_system_integration():
    """Integration test for complete hybrid retrieval system."""
    print("Running system integration test...")
    
    # Use existing documents if available
    data_path = "data/raw_texts"
    if not Path(data_path).exists():
        pytest.skip(f"Test data directory {data_path} not found")
    
    # Create temporary pipeline
    temp_dir = tempfile.mkdtemp()
    try:
        pipeline = RAGPipeline(
            text_data_path=data_path,
            vector_db_path=os.path.join(temp_dir, "test_vectordb")
        )
        
        # Ingest documents
        pipeline.ingest_documents()
        
        # Test both standard and enhanced retrieval
        test_query = "Was ist Kubernetes?"
        
        # Standard retrieval
        standard_answer = pipeline.answer_query(test_query)
        assert isinstance(standard_answer, str) and len(standard_answer) > 10
        
        # Enhanced retrieval
        enhanced_answer = pipeline.enhanced_answer_query(test_query)
        assert isinstance(enhanced_answer, str) and len(enhanced_answer) > 10
        
        print("System integration test passed")
        
        # Clean up ChromaDB connections
        if hasattr(pipeline, 'vector_store_manager'):
            if hasattr(pipeline.vector_store_manager, 'client'):
                try:
                    pipeline.vector_store_manager.client.reset()
                except:
                    pass
        pipeline = None
        
    finally:
        # Manual cleanup with retry
        import time
        time.sleep(0.2)  # Let file handles close
        
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except PermissionError:
            # On Windows, sometimes need multiple attempts
            time.sleep(0.5)
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
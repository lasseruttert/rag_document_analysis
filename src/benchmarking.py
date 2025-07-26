#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance Benchmarking System for Enhanced Hybrid Retrieval.

This module provides comprehensive benchmarking capabilities to measure
and compare the performance of different retrieval methods.
"""

import time
import statistics
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class BenchmarkResult:
    """Single benchmark result for a query."""
    query: str
    query_type: str
    expected_files: List[str]
    method: str
    response_time_ms: float
    top_k_accuracy: Dict[int, float]  # {1: 0.5, 3: 0.8, 5: 1.0}
    relevance_scores: List[float]
    semantic_weight: float = 0.0
    keyword_weight: float = 0.0
    total_results: int = 0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    name: str
    description: str
    timestamp: str
    results: List[BenchmarkResult]
    summary_stats: Dict[str, Any]

class PerformanceBenchmarker:
    """
    Comprehensive benchmarking system for RAG retrieval methods.
    
    Features:
    - Query-type-specific benchmarking (technical/question/balanced)
    - Comparative analysis (standard vs hybrid retrieval)
    - German language specific tests
    - Performance regression detection
    - Detailed reporting with visualizations
    """
    
    def __init__(self, pipeline, output_dir: str = "benchmarks/results"):
        """
        Initialize benchmarker with RAG pipeline.
        
        Args:
            pipeline: RAGPipeline instance
            output_dir: Directory to save benchmark results
        """
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test dataset - realistic queries based on our documents
        self.test_queries = self._create_test_dataset()
    
    def _create_test_dataset(self) -> List[Dict[str, Any]]:
        """Create comprehensive test dataset with ground truth."""
        return [
            # Question-type queries (should favor semantic retrieval)
            {
                'query': 'Was ist ein Kubernetes Pod?',
                'query_type': 'question',
                'expected_files': ['kubernetes_basics.txt'],
                'expected_keywords': ['pod', 'kubernetes', 'container'],
                'complexity': 'simple'
            },
            {
                'query': 'Wie funktioniert Container-Orchestrierung in Kubernetes?',
                'query_type': 'question',
                'expected_files': ['kubernetes_basics.txt'],
                'expected_keywords': ['orchestrierung', 'container', 'kubernetes'],
                'complexity': 'medium'
            },
            {
                'query': 'Warum sollte ich virtuelle Umgebungen in Python verwenden?',
                'query_type': 'question',
                'expected_files': ['python_best_practices.txt'],
                'expected_keywords': ['virtuelle', 'umgebungen', 'python'],
                'complexity': 'medium'
            },
            
            # Technical queries (should favor keyword retrieval)
            {
                'query': 'kubectl get pods',
                'query_type': 'technical',
                'expected_files': ['kubernetes_basics.txt'],
                'expected_keywords': ['kubectl', 'pods', 'get'],
                'complexity': 'simple'
            },
            {
                'query': 'PEP 8 Stilrichtlinien Python',
                'query_type': 'technical',
                'expected_files': ['python_best_practices.txt'],
                'expected_keywords': ['pep', '8', 'python', 'stil'],
                'complexity': 'simple'
            },
            {
                'query': 'Docker Compose deployment configuration',
                'query_type': 'technical',
                'expected_files': ['kubernetes_basics.txt'],  # May also find in k8s context
                'expected_keywords': ['docker', 'compose', 'deployment'],
                'complexity': 'medium'
            },
            
            # Balanced queries (should use adaptive weighting)
            {
                'query': 'Python Best Practices Tutorial',
                'query_type': 'balanced',
                'expected_files': ['python_best_practices.txt'],
                'expected_keywords': ['python', 'best', 'practices'],
                'complexity': 'simple'
            },
            {
                'query': 'Kubernetes Container Management Guide',
                'query_type': 'balanced',
                'expected_files': ['kubernetes_basics.txt'],
                'expected_keywords': ['kubernetes', 'container', 'management'],
                'complexity': 'medium'
            },
            
            # German-specific language tests
            {
                'query': 'Anwendungscontainer mit Kubernetes verwalten',
                'query_type': 'balanced',
                'expected_files': ['kubernetes_basics.txt'],
                'expected_keywords': ['anwendung', 'container', 'kubernetes'],
                'complexity': 'medium'
            },
            {
                'query': 'Entwicklungsumgebung für Python-Projekte einrichten',
                'query_type': 'balanced',
                'expected_files': ['python_best_practices.txt'],
                'expected_keywords': ['entwicklung', 'python', 'projekt'],
                'complexity': 'complex'
            },
            
            # Edge cases and challenging queries
            {
                'query': 'Service Discovery Kubernetes',
                'query_type': 'technical',
                'expected_files': ['kubernetes_basics.txt'],
                'expected_keywords': ['service', 'discovery', 'kubernetes'],
                'complexity': 'medium'
            },
            {
                'query': 'Was sind die Unterschiede zwischen Docker und Kubernetes?',
                'query_type': 'question',
                'expected_files': ['kubernetes_basics.txt'],
                'expected_keywords': ['docker', 'kubernetes', 'unterschiede'],
                'complexity': 'complex'
            }
        ]
    
    def run_comprehensive_benchmark(self, methods: List[str] = None) -> BenchmarkSuite:
        """
        Run comprehensive benchmark comparing different retrieval methods.
        
        Args:
            methods: List of methods to test ['standard', 'hybrid', 'semantic_only', 'keywords_only']
        
        Returns:
            BenchmarkSuite with complete results
        """
        if methods is None:
            methods = ['standard', 'hybrid']
        
        print("🚀 Starting Comprehensive RAG Performance Benchmark")
        print(f"📊 Testing {len(self.test_queries)} queries with {len(methods)} methods")
        print("=" * 60)
        
        all_results = []
        
        # Ensure pipeline is ready
        if self.pipeline.collection.count() == 0:
            print("📋 Ingesting documents for benchmark...")
            self.pipeline.ingest_documents()
        
        for method in methods:
            print(f"\n🔍 Testing method: {method.upper()}")
            print("-" * 40)
            
            method_results = self._benchmark_method(method)
            all_results.extend(method_results)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(all_results)
        
        # Create benchmark suite
        suite = BenchmarkSuite(
            name="RAG Hybrid Retrieval Benchmark",
            description="Comprehensive comparison of retrieval methods with German language support",
            timestamp=datetime.now().isoformat(),
            results=all_results,
            summary_stats=summary_stats
        )
        
        # Save results
        self._save_benchmark_results(suite)
        
        # Print summary
        self._print_benchmark_summary(suite)
        
        return suite
    
    def _benchmark_method(self, method: str) -> List[BenchmarkResult]:
        """Benchmark a specific retrieval method."""
        results = []
        
        for i, test_case in enumerate(self.test_queries, 1):
            query = test_case['query']
            expected_files = test_case['expected_files']
            query_type = test_case['query_type']
            
            print(f"  {i:2d}/{len(self.test_queries)} | {query_type:9s} | {query[:50]:<50}")
            
            # Run retrieval and measure time
            start_time = time.perf_counter()
            
            try:
                if method == 'standard':
                    retrieved_chunks = self.pipeline.retriever.retrieve(query, top_k=10)
                    semantic_weight = 1.0
                    keyword_weight = 0.0
                elif method == 'hybrid':
                    # Setup enhanced retriever if needed
                    if not hasattr(self.pipeline, 'enhanced_retriever') or self.pipeline.enhanced_retriever is None:
                        self.pipeline._setup_enhanced_retriever()
                    
                    retrieved_chunks = self.pipeline.enhanced_retriever.hybrid_retrieve(query, top_k=10)
                    semantic_weight = retrieved_chunks[0].get('semantic_weight', 0.6) if retrieved_chunks else 0.6
                    keyword_weight = retrieved_chunks[0].get('keyword_weight', 0.4) if retrieved_chunks else 0.4
                elif method == 'semantic_only':
                    retrieved_chunks = self.pipeline.retriever.retrieve(query, top_k=10)
                    semantic_weight = 1.0
                    keyword_weight = 0.0
                elif method == 'keywords_only':
                    # Simulate keyword-only by using BM25 if available
                    if hasattr(self.pipeline, 'enhanced_retriever') and self.pipeline.enhanced_retriever:
                        retrieved_chunks = self.pipeline.enhanced_retriever.bm25_retriever.retrieve(query, top_k=10)
                    else:
                        retrieved_chunks = self.pipeline.retriever.retrieve(query, top_k=10)
                    semantic_weight = 0.0
                    keyword_weight = 1.0
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                response_time_ms = (time.perf_counter() - start_time) * 1000
                
                # Evaluate results
                top_k_accuracy = self._calculate_accuracy(retrieved_chunks, expected_files)
                relevance_scores = self._calculate_relevance_scores(retrieved_chunks, test_case)
                
                result = BenchmarkResult(
                    query=query,
                    query_type=query_type,
                    expected_files=expected_files,
                    method=method,
                    response_time_ms=round(response_time_ms, 2),
                    top_k_accuracy=top_k_accuracy,
                    relevance_scores=relevance_scores,
                    semantic_weight=semantic_weight,
                    keyword_weight=keyword_weight,
                    total_results=len(retrieved_chunks)
                )
                
                results.append(result)
                
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
                # Create failed result
                result = BenchmarkResult(
                    query=query,
                    query_type=query_type,
                    expected_files=expected_files,
                    method=method,
                    response_time_ms=0.0,
                    top_k_accuracy={},
                    relevance_scores=[],
                    total_results=0
                )
                results.append(result)
        
        return results
    
    def _calculate_accuracy(self, retrieved_chunks: List[Dict], expected_files: List[str]) -> Dict[int, float]:
        """Calculate top-K accuracy for different K values."""
        if not retrieved_chunks:
            return {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}
        
        # Get retrieved filenames
        retrieved_files = [chunk['metadata']['filename'] for chunk in retrieved_chunks]
        
        accuracy = {}
        for k in [1, 3, 5, 10]:
            top_k_files = retrieved_files[:k]
            hits = sum(1 for expected in expected_files if any(expected in retrieved for retrieved in top_k_files))
            accuracy[k] = hits / len(expected_files) if expected_files else 0.0
        
        return accuracy
    
    def _calculate_relevance_scores(self, retrieved_chunks: List[Dict], test_case: Dict) -> List[float]:
        """Calculate relevance scores for retrieved chunks."""
        if not retrieved_chunks:
            return []
        
        expected_keywords = test_case.get('expected_keywords', [])
        relevance_scores = []
        
        for chunk in retrieved_chunks:
            content = chunk['content'].lower()
            
            # Simple relevance based on keyword presence and distance/score
            keyword_score = sum(1 for keyword in expected_keywords if keyword.lower() in content) / len(expected_keywords) if expected_keywords else 0.5
            
            # Get retrieval score (distance for semantic, bm25_score for keyword, hybrid_score for hybrid)
            if 'hybrid_score' in chunk:
                retrieval_score = chunk['hybrid_score']
            elif 'bm25_score' in chunk:
                retrieval_score = min(chunk['bm25_score'] / 10.0, 1.0)  # Normalize BM25 score
            else:
                retrieval_score = max(0.0, 1.0 - chunk.get('distance', 1.0))  # Convert distance to score
            
            # Combined relevance score
            relevance = (keyword_score * 0.4 + retrieval_score * 0.6)
            relevance_scores.append(round(relevance, 3))
        
        return relevance_scores
    
    def _calculate_summary_stats(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate summary statistics across all results."""
        if not results:
            return {}
        
        # Group by method
        methods = {}
        for result in results:
            if result.method not in methods:
                methods[result.method] = []
            methods[result.method].append(result)
        
        summary = {}
        
        for method, method_results in methods.items():
            if not method_results:
                continue
            
            # Response times
            response_times = [r.response_time_ms for r in method_results if r.response_time_ms > 0]
            
            # Accuracy metrics
            top1_accuracies = [r.top_k_accuracy.get(1, 0.0) for r in method_results]
            top3_accuracies = [r.top_k_accuracy.get(3, 0.0) for r in method_results]
            top5_accuracies = [r.top_k_accuracy.get(5, 0.0) for r in method_results]
            
            # Average relevance scores
            all_relevance_scores = []
            for r in method_results:
                all_relevance_scores.extend(r.relevance_scores)
            
            method_stats = {
                'total_queries': len(method_results),
                'avg_response_time_ms': round(statistics.mean(response_times), 2) if response_times else 0.0,
                'median_response_time_ms': round(statistics.median(response_times), 2) if response_times else 0.0,
                'p95_response_time_ms': round(np.percentile(response_times, 95), 2) if response_times else 0.0,
                'top1_accuracy': round(statistics.mean(top1_accuracies), 3),
                'top3_accuracy': round(statistics.mean(top3_accuracies), 3),
                'top5_accuracy': round(statistics.mean(top5_accuracies), 3),
                'avg_relevance_score': round(statistics.mean(all_relevance_scores), 3) if all_relevance_scores else 0.0,
                'successful_queries': len([r for r in method_results if r.response_time_ms > 0])
            }
            
            # Query type breakdown
            query_types = {}
            for result in method_results:
                qt = result.query_type
                if qt not in query_types:
                    query_types[qt] = {'count': 0, 'avg_top3_accuracy': 0.0, 'avg_response_time': 0.0}
                
                query_types[qt]['count'] += 1
                query_types[qt]['avg_top3_accuracy'] += result.top_k_accuracy.get(3, 0.0)
                query_types[qt]['avg_response_time'] += result.response_time_ms
            
            for qt in query_types:
                count = query_types[qt]['count']
                query_types[qt]['avg_top3_accuracy'] = round(query_types[qt]['avg_top3_accuracy'] / count, 3)
                query_types[qt]['avg_response_time'] = round(query_types[qt]['avg_response_time'] / count, 2)
            
            method_stats['query_type_breakdown'] = query_types
            summary[method] = method_stats
        
        return summary
    
    def _save_benchmark_results(self, suite: BenchmarkSuite):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = self.output_dir / f"benchmark_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            # Convert dataclasses to dict for JSON serialization
            suite_dict = asdict(suite)
            json.dump(suite_dict, f, indent=2, ensure_ascii=False)
        
        # Save CSV for detailed analysis
        csv_file = self.output_dir / f"benchmark_{timestamp}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if suite.results:
                fieldnames = ['timestamp', 'query', 'query_type', 'method', 'response_time_ms', 
                             'top1_accuracy', 'top3_accuracy', 'top5_accuracy', 'avg_relevance_score',
                             'semantic_weight', 'keyword_weight', 'total_results']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in suite.results:
                    row = {
                        'timestamp': result.timestamp,
                        'query': result.query,
                        'query_type': result.query_type,
                        'method': result.method,
                        'response_time_ms': result.response_time_ms,
                        'top1_accuracy': result.top_k_accuracy.get(1, 0.0),
                        'top3_accuracy': result.top_k_accuracy.get(3, 0.0),
                        'top5_accuracy': result.top_k_accuracy.get(5, 0.0),
                        'avg_relevance_score': statistics.mean(result.relevance_scores) if result.relevance_scores else 0.0,
                        'semantic_weight': result.semantic_weight,
                        'keyword_weight': result.keyword_weight,
                        'total_results': result.total_results
                    }
                    writer.writerow(row)
        
        print(f"\n💾 Results saved:")
        print(f"   📄 JSON: {json_file}")
        print(f"   📊 CSV:  {csv_file}")
    
    def _print_benchmark_summary(self, suite: BenchmarkSuite):
        """Print comprehensive benchmark summary."""
        print("\n" + "=" * 80)
        print("🎯 BENCHMARK SUMMARY REPORT")
        print("=" * 80)
        
        for method, stats in suite.summary_stats.items():
            print(f"\n📊 Method: {method.upper()}")
            print("-" * 50)
            print(f"Total Queries:           {stats['total_queries']}")
            print(f"Successful Queries:      {stats['successful_queries']}")
            print(f"Avg Response Time:       {stats['avg_response_time_ms']:.1f}ms")
            print(f"95th Percentile Time:    {stats['p95_response_time_ms']:.1f}ms")
            print(f"Top-1 Accuracy:          {stats['top1_accuracy']:.1%}")
            print(f"Top-3 Accuracy:          {stats['top3_accuracy']:.1%}")
            print(f"Top-5 Accuracy:          {stats['top5_accuracy']:.1%}")
            print(f"Avg Relevance Score:     {stats['avg_relevance_score']:.3f}")
            
            print(f"\n  📋 Query Type Breakdown:")
            for qt, qt_stats in stats.get('query_type_breakdown', {}).items():
                print(f"    {qt:12s}: {qt_stats['count']:2d} queries, "
                      f"Top-3: {qt_stats['avg_top3_accuracy']:.1%}, "
                      f"Time: {qt_stats['avg_response_time']:.1f}ms")
        
        # Comparison if multiple methods
        if len(suite.summary_stats) > 1:
            print(f"\n🆚 METHOD COMPARISON")
            print("-" * 50)
            
            methods = list(suite.summary_stats.keys())
            if 'standard' in methods and 'hybrid' in methods:
                std_stats = suite.summary_stats['standard']
                hyb_stats = suite.summary_stats['hybrid']
                
                top3_improvement = ((hyb_stats['top3_accuracy'] - std_stats['top3_accuracy']) / std_stats['top3_accuracy'] * 100) if std_stats['top3_accuracy'] > 0 else 0
                time_change = ((hyb_stats['avg_response_time_ms'] - std_stats['avg_response_time_ms']) / std_stats['avg_response_time_ms'] * 100) if std_stats['avg_response_time_ms'] > 0 else 0
                
                print(f"Top-3 Accuracy Improvement: {top3_improvement:+.1f}%")
                print(f"Response Time Change:       {time_change:+.1f}%")
                print(f"Relevance Score Improvement: {(hyb_stats['avg_relevance_score'] - std_stats['avg_relevance_score']):.3f}")
        
        print(f"\n📅 Benchmark completed at: {suite.timestamp}")
        print("=" * 80)

def run_performance_benchmark():
    """Convenience function to run performance benchmark."""
    import sys
    import os
    
    # Fix encoding issues on Windows
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.pipeline import RAGPipeline
    
    print("🚀 Initializing RAG Pipeline for Benchmarking...")
    pipeline = RAGPipeline()
    
    benchmarker = PerformanceBenchmarker(pipeline)
    
    # Run comprehensive benchmark
    results = benchmarker.run_comprehensive_benchmark(['standard', 'hybrid'])
    
    return results

if __name__ == '__main__':
    results = run_performance_benchmark()
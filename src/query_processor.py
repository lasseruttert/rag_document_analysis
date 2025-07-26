#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Query Enhancement and Preprocessing System for RAG Document Analysis.

This module provides intelligent query processing capabilities including
query expansion, spell checking, intent detection, and normalization.
"""

import re
import logging
import difflib
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
try:
    from src.config import get_config
except ImportError:
    # Fallback for testing
    def get_config():
        return None

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Query intent types for better processing."""
    QUESTION = "question"
    COMMAND = "command" 
    SEARCH = "search"
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    COMPARISON = "comparison"
    UNKNOWN = "unknown"

@dataclass
class QueryAnalysisResult:
    """Result of query analysis and preprocessing."""
    original_query: str
    processed_query: str
    expanded_query: str
    corrected_query: str
    intent: QueryIntent
    confidence: float
    suggestions: List[str]
    extracted_keywords: List[str]
    language: str = "de"

class GermanQueryExpander:
    """German-specific query expansion with domain knowledge."""
    
    def __init__(self):
        # German technical synonyms and expansions
        self.synonyms = {
            # Kubernetes terms
            "pod": ["container", "workload", "process"],
            "service": ["dienst", "svc", "endpoint"],
            "deployment": ["bereitstellung", "deploy", "rollout"],
            "cluster": ["verbund", "system", "infrastruktur"],
            "node": ["knoten", "server", "host", "maschine"],
            "namespace": ["namensraum", "bereich", "umgebung"],
            
            # Python terms
            "function": ["funktion", "methode", "def"],
            "class": ["klasse", "objekt", "typ"],
            "variable": ["var", "wert", "parameter"],
            "import": ["importieren", "laden", "einbinden"],
            "exception": ["fehler", "error", "ausnahme"],
            "module": ["modul", "paket", "bibliothek"],
            
            # General technical terms
            "config": ["konfiguration", "einstellung", "parameter"],
            "debug": ["debuggen", "fehlersuche", "troubleshoot"],
            "install": ["installieren", "einrichten", "setup"],
            "update": ["aktualisieren", "updaten", "erneuern"],
            "delete": ["löschen", "entfernen", "remove"],
            "create": ["erstellen", "anlegen", "erzeugen"],
            
            # Documentation terms
            "beispiel": ["example", "demo", "sample"],
            "tutorial": ["anleitung", "guide", "howto"],
            "best practice": ["bewährte verfahren", "empfehlung", "standard"],
            "troubleshooting": ["fehlerbehebung", "problemlösung", "debug"]
        }
        
        # Common German compound word patterns
        self.compound_patterns = [
            r'(\w+)management',
            r'(\w+)konfiguration', 
            r'(\w+)installation',
            r'(\w+)dokumentation',
            r'(\w+)implementierung'
        ]
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with German synonyms and related terms.
        
        Args:
            query: Original query string
            
        Returns:
            List of expanded query variations
        """
        expansions = [query.lower()]
        words = re.findall(r'\w+', query.lower())
        
        # Add synonyms for each word
        for word in words:
            if word in self.synonyms:
                for synonym in self.synonyms[word]:
                    expanded = query.lower().replace(word, synonym)
                    if expanded not in expansions:
                        expansions.append(expanded)
        
        # Handle compound words
        for pattern in self.compound_patterns:
            matches = re.findall(pattern, query.lower())
            for match in matches:
                # Split compound word
                compound_expansion = f"{match} management" if "management" in pattern else f"{match} konfiguration"
                expansions.append(compound_expansion)
        
        return expansions[:5]  # Limit to top 5 expansions

class GermanSpellChecker:
    """German spell checking and correction for technical terms."""
    
    def __init__(self):
        # German technical vocabulary
        self.vocabulary = {
            # Kubernetes
            "kubernetes", "pod", "service", "deployment", "cluster", "node", 
            "namespace", "container", "docker", "helm", "kubectl", "ingress",
            
            # Python
            "python", "function", "class", "import", "exception", "module",
            "variable", "list", "dict", "string", "integer", "boolean",
            
            # German technical terms
            "konfiguration", "installation", "implementierung", "dokumentation",
            "bereitstellung", "verwaltung", "entwicklung", "programmierung",
            "fehlerbehebung", "troubleshooting", "debugging", "monitoring",
            
            # Common verbs
            "erstellen", "löschen", "aktualisieren", "installieren", "konfigurieren",
            "debuggen", "testen", "deployen", "skalieren", "überwachen"
        }
        
        self.min_similarity = 0.6
    
    def correct_spelling(self, query: str) -> Tuple[str, List[str]]:
        """
        Correct spelling errors in query.
        
        Args:
            query: Original query
            
        Returns:
            Tuple of (corrected_query, suggestions)
        """
        words = re.findall(r'\w+', query.lower())
        corrected_words = []
        suggestions = []
        
        for word in words:
            if len(word) < 3:  # Skip very short words
                corrected_words.append(word)
                continue
                
            if word in self.vocabulary:
                corrected_words.append(word)
            else:
                # Find best match
                matches = difflib.get_close_matches(
                    word, self.vocabulary, 
                    n=3, cutoff=self.min_similarity
                )
                
                if matches:
                    best_match = matches[0]
                    corrected_words.append(best_match)
                    if word != best_match:
                        suggestions.append(f"'{word}' -> '{best_match}'")
                else:
                    corrected_words.append(word)
        
        # Reconstruct query with corrected words
        corrected_query = query
        for i, word in enumerate(re.findall(r'\w+', query)):
            if i < len(corrected_words):
                corrected_query = corrected_query.replace(word, corrected_words[i], 1)
        
        return corrected_query, suggestions

class QueryIntentDetector:
    """Detect intent and type of user queries."""
    
    def __init__(self):
        # German question patterns
        self.question_patterns = [
            r'^(was|wie|wo|wann|warum|welche|welcher|welches|wer)\s',
            r'\?$',
            r'^(erkläre|beschreibe|zeige)',
            r'(was ist|wie funktioniert|was bedeutet)'
        ]
        
        # Command patterns
        self.command_patterns = [
            r'^(erstelle|lösche|installiere|konfiguriere|starte|stoppe)',
            r'^(zeige mir|gib mir|liste)',
            r'^(führe aus|execute|run)',
            r'(schritt für schritt|anleitung|howto)'
        ]
        
        # Comparison patterns  
        self.comparison_patterns = [
            r'(unterschied|vergleich|vs\.|versus)',
            r'(besser|schlechter|vor- und nachteile)',
            r'(welche.*besser|was ist der unterschied)'
        ]
        
        # Procedural patterns
        self.procedural_patterns = [
            r'(wie.*mache|wie.*erstelle|wie.*installiere)',
            r'(schritt.*schritt|tutorial|anleitung)',
            r'(setup|installation|konfiguration)'
        ]
    
    def detect_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """
        Detect the intent of a query.
        
        Args:
            query: Query string to analyze
            
        Returns:
            Tuple of (intent, confidence_score)
        """
        query_lower = query.lower().strip()
        
        # Check for question patterns
        for pattern in self.question_patterns:
            if re.search(pattern, query_lower):
                return QueryIntent.QUESTION, 0.9
        
        # Check for command patterns
        for pattern in self.command_patterns:
            if re.search(pattern, query_lower):
                return QueryIntent.COMMAND, 0.8
        
        # Check for comparison patterns
        for pattern in self.comparison_patterns:
            if re.search(pattern, query_lower):
                return QueryIntent.COMPARISON, 0.8
        
        # Check for procedural patterns
        for pattern in self.procedural_patterns:
            if re.search(pattern, query_lower):
                return QueryIntent.PROCEDURAL, 0.8
        
        # Default classification based on length and structure
        if len(query_lower.split()) <= 3:
            return QueryIntent.SEARCH, 0.6
        else:
            return QueryIntent.FACTUAL, 0.5

class QueryProcessor:
    """
    Comprehensive query processing and enhancement system.
    
    Provides intelligent query preprocessing including expansion,
    spell checking, intent detection, and normalization.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize query processor with configuration.
        
        Args:
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()
        
        # Initialize components
        self.expander = GermanQueryExpander()
        self.spell_checker = GermanSpellChecker()
        self.intent_detector = QueryIntentDetector()
        
        # Processing options
        self.enable_expansion = True
        self.enable_spell_check = True
        self.enable_intent_detection = True
        self.enable_normalization = True
    
    def preprocess(self, query: str) -> QueryAnalysisResult:
        """
        Comprehensive query preprocessing and analysis.
        
        Args:
            query: Raw user query
            
        Returns:
            QueryAnalysisResult with all processing results
        """
        if not query or not query.strip():
            return QueryAnalysisResult(
                original_query="",
                processed_query="",
                expanded_query="",
                corrected_query="",
                intent=QueryIntent.UNKNOWN,
                confidence=0.0,
                suggestions=[],
                extracted_keywords=[]
            )
        
        original_query = query.strip()
        
        # Step 1: Normalize query
        normalized_query = self._normalize_query(original_query)
        
        # Step 2: Spell checking
        corrected_query = normalized_query
        suggestions = []
        if self.enable_spell_check:
            corrected_query, suggestions = self.spell_checker.correct_spelling(normalized_query)
        
        # Step 3: Query expansion
        expanded_query = corrected_query
        if self.enable_expansion:
            expansions = self.expander.expand_query(corrected_query)
            expanded_query = " OR ".join(expansions)
        
        # Step 4: Intent detection
        intent = QueryIntent.UNKNOWN
        confidence = 0.0
        if self.enable_intent_detection:
            intent, confidence = self.intent_detector.detect_intent(corrected_query)
        
        # Step 5: Extract keywords
        keywords = self._extract_keywords(corrected_query)
        
        return QueryAnalysisResult(
            original_query=original_query,
            processed_query=corrected_query,
            expanded_query=expanded_query,
            corrected_query=corrected_query,
            intent=intent,
            confidence=confidence,
            suggestions=suggestions,
            extracted_keywords=keywords,
            language="de"
        )
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query text.
        
        Args:
            query: Raw query string
            
        Returns:
            Normalized query string
        """
        if not self.enable_normalization:
            return query
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', query.strip())
        
        # Remove special characters except meaningful ones
        normalized = re.sub(r'[^\w\s\-\.\?\!]', ' ', normalized)
        
        # Normalize German umlauts if needed
        umlaut_map = {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss'}
        for umlaut, replacement in umlaut_map.items():
            normalized = normalized.replace(umlaut, replacement)
        
        return normalized.strip()
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query.
        
        Args:
            query: Processed query string
            
        Returns:
            List of extracted keywords
        """
        # Simple keyword extraction based on word length and frequency
        words = re.findall(r'\w+', query.lower())
        
        # Filter out common German stop words
        german_stopwords = {
            'der', 'die', 'das', 'und', 'oder', 'aber', 'ist', 'sind', 'war', 'waren',
            'ein', 'eine', 'einen', 'einem', 'einer', 'eines', 'mit', 'von', 'zu', 'auf',
            'in', 'an', 'bei', 'für', 'über', 'unter', 'durch', 'gegen', 'ohne', 'um',
            'wie', 'was', 'wo', 'wann', 'warum', 'welche', 'welcher', 'welches'
        }
        
        keywords = []
        for word in words:
            if (len(word) >= 3 and 
                word not in german_stopwords and
                not word.isdigit()):
                keywords.append(word)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(keywords))
    
    def suggest_queries(self, partial_query: str, limit: int = 5) -> List[str]:
        """
        Suggest query completions based on partial input.
        
        Args:
            partial_query: Partial query string
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested query completions
        """
        if len(partial_query) < 2:
            return []
        
        # Common query patterns for suggestions
        common_patterns = [
            "Was ist {term}?",
            "Wie funktioniert {term}?", 
            "Wie erstelle ich {term}?",
            "Wie konfiguriere ich {term}?",
            "Was sind die Vorteile von {term}?",
            "Unterschied zwischen {term} und",
            "{term} Best Practices",
            "{term} Tutorial",
            "{term} Beispiele",
            "{term} Troubleshooting"
        ]
        
        suggestions = []
        partial_lower = partial_query.lower()
        
        # Look for matching vocabulary terms
        matching_terms = [
            term for term in self.spell_checker.vocabulary 
            if term.startswith(partial_lower)
        ]
        
        # Generate suggestions based on patterns
        for term in matching_terms[:3]:  # Top 3 matching terms
            for pattern in common_patterns[:limit]:
                suggestion = pattern.format(term=term)
                if suggestion not in suggestions:
                    suggestions.append(suggestion)
                    if len(suggestions) >= limit:
                        break
            if len(suggestions) >= limit:
                break
        
        return suggestions

# Test and example usage
if __name__ == '__main__':
    print("Testing Query Processing System...")
    
    # Initialize processor
    processor = QueryProcessor()
    
    # Test queries
    test_queries = [
        "Was ist ein Pod in Kubernetes?",
        "wie erstele ich ein deployment",  # Typo: erstele -> erstelle
        "kubectl get pods",
        "python funktion definieren",
        "docker vs kubernetes unterschied",
        "installiere nginx",
        "kubernetes troubleshooting"
    ]
    
    print("\nQuery Processing Results:")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Original Query: '{query}'")
        
        result = processor.preprocess(query)
        
        print(f"   Processed: '{result.processed_query}'")
        print(f"   Intent: {result.intent.value} (confidence: {result.confidence:.2f})")
        print(f"   Keywords: {result.extracted_keywords}")
        
        if result.suggestions:
            print(f"   Spelling suggestions: {result.suggestions}")
        
        if result.expanded_query != result.processed_query:
            print(f"   Expanded: '{result.expanded_query[:100]}...'")
    
    # Test query suggestions
    print(f"\n\nQuery Suggestions for 'kubern':")
    suggestions = processor.suggest_queries("kubern")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"   {i}. {suggestion}")
    
    print("\nQuery Processing System test completed!")
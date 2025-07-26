from typing import List, Dict, Tuple, Any
import re
import logging
import hashlib
import pickle
import os
import tempfile
from pathlib import Path
from src.embeddings import EmbeddingManager
from src.vectorstore import VectorStoreManager
from src.constants import (
    COMBINED_STOPWORDS, TECHNICAL_PATTERNS, CODE_KEYWORDS, QUESTION_PATTERNS,
    DEFAULT_SEMANTIC_WEIGHT, DEFAULT_KEYWORD_WEIGHT, HYBRID_BOOST_FACTOR,
    BM25_K1, BM25_B, BM25_MIN_TOKEN_LENGTH
)

logger = logging.getLogger(__name__)

# Hybrid Retrieval Dependencies
from rank_bm25 import BM25Okapi
from nltk.stem import SnowballStemmer
import nltk

class Retriever:
    def __init__(self, embedding_manager: EmbeddingManager, vector_store_manager: VectorStoreManager):
        """
        Initialisiert den Retriever.

        Args:
            embedding_manager: Eine Instanz des EmbeddingManagers.
            vector_store_manager: Eine Instanz des VectorStoreManagers.
        """
        self.embedding_manager = embedding_manager
        self.vector_store_manager = vector_store_manager

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, any]]:
        """
        Führt eine semantische Suche für eine gegebene Anfrage durch.

        Args:
            query: Die Suchanfrage des Benutzers.
            top_k: Die Anzahl der zurückzugebenden relevantesten Dokumente.

        Returns:
            Eine Liste der relevantesten Dokumenten-Chunks.
        """
        if not self.vector_store_manager.collection:
            raise ValueError("Vector store collection not initialized.")

        # 1. Query-Embedding generieren
        logger.debug(f"Generating embedding for query: '{query}'")
        query_embedding = self.embedding_manager.model.encode(query, convert_to_numpy=True)

        # 2. Ähnlichkeitssuche in der Vektordatenbank
        logger.debug(f"Querying vector store for top {top_k} results...")
        results = self.vector_store_manager.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )

        # 3. Ergebnisse formatieren
        retrieved_chunks = []
        for i in range(len(results['ids'][0])):
            retrieved_chunks.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return retrieved_chunks


class QueryAnalyzer:
    """Analysiert Query-Typ für adaptive Gewichtung."""
    
    def __init__(self):
        self.technical_patterns = TECHNICAL_PATTERNS
        self.code_keywords = CODE_KEYWORDS
    
    def analyze_query_type(self, query: str) -> Dict[str, float]:
        """
        Bestimmt Query-Typ und returniert Gewichtungen.
        
        Returns:
            Dict mit 'semantic_weight' und 'keyword_weight' (sum = 1.0)
        """
        query_lower = query.lower()
        
        # Technische Pattern-Erkennung
        technical_score = 0.0
        for pattern in self.technical_patterns:
            if re.search(pattern, query):
                technical_score += 0.3
        
        # Code-Keyword-Erkennung (höhere Gewichtung für mehrere Keywords)
        keyword_matches = sum(1 for keyword in self.code_keywords 
                             if keyword in query_lower)
        code_score = min(keyword_matches * 0.3, 1.0)  # 0.3 per keyword, max 1.0
        
        # Frage-Pattern (semantisch orientiert)
        question_score = sum(0.25 for pattern in QUESTION_PATTERNS 
                           if re.search(pattern, query_lower, re.IGNORECASE))
        
        # Adaptive Gewichtung basierend auf Scores
        technical_total = min(technical_score + code_score, 1.0)
        
        if technical_total > 0.5:
            # Technische Query → mehr Keyword-Gewicht
            keyword_weight = 0.6 + (technical_total - 0.5) * 0.3
            semantic_weight = 1.0 - keyword_weight
        elif question_score > 0.5:
            # Frage-Query → mehr Semantic-Gewicht
            semantic_weight = 0.7 + question_score * 0.2
            keyword_weight = 1.0 - semantic_weight
        else:
            # Balanced Query
            semantic_weight = DEFAULT_SEMANTIC_WEIGHT
            keyword_weight = DEFAULT_KEYWORD_WEIGHT
        
        return {
            'semantic_weight': round(semantic_weight, 2),
            'keyword_weight': round(keyword_weight, 2),
            'query_type': 'technical' if technical_total > 0.5 else 
                         'question' if question_score > 0.5 else 'balanced'
        }


class GermanBM25Retriever:
    """BM25-Retriever mit deutscher Sprachoptimierung und Caching."""
    
    def __init__(self, documents: List[Dict[str, Any]], cache_dir: str = None, enable_cache: bool = True):
        self.documents = documents
        self.stemmer = SnowballStemmer('german')
        self.enable_cache = enable_cache
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), 'rag_bm25_cache')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Generate cache key based on document content
        self.cache_key = self._generate_cache_key(documents)
        self.cache_file = self.cache_dir / f"bm25_{self.cache_key}.pkl"
        
        # Try to load from cache or build new
        if self.enable_cache and self._load_from_cache():
            logger.info(f"Loaded BM25 model from cache: {self.cache_file}")
        else:
            logger.info("Building new BM25 model...")
            self._build_bm25_model()
            if self.enable_cache:
                self._save_to_cache()
                logger.info(f"Saved BM25 model to cache: {self.cache_file}")
    
    def _generate_cache_key(self, documents: List[Dict[str, Any]]) -> str:
        """Generate a unique cache key based on document content."""
        content_hash = hashlib.md5()
        
        # Create a stable hash from document content and metadata
        for doc in sorted(documents, key=lambda x: x.get('id', '')):
            content = doc.get('content', '')
            chunk_id = doc.get('metadata', {}).get('chunk_id', '')
            combined = f"{chunk_id}:{content}"
            content_hash.update(combined.encode('utf-8'))
        
        return content_hash.hexdigest()[:16]  # Use first 16 chars
    
    def _build_bm25_model(self):
        """Build BM25 model from documents."""
        # Prepare BM25 corpus
        self.tokenized_corpus = [self._preprocess_text(doc['content']) 
                                for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def _save_to_cache(self) -> bool:
        """Save BM25 model and tokenized corpus to cache."""
        try:
            cache_data = {
                'tokenized_corpus': self.tokenized_corpus,
                'bm25_idf': self.bm25.idf,
                'bm25_doc_freqs': self.bm25.doc_freqs,
                'bm25_doc_len': self.bm25.doc_len,
                'bm25_avgdl': self.bm25.avgdl,
                'documents_count': len(self.documents),
                'cache_version': '1.0'
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            return True
        except Exception as e:
            logger.warning(f"Failed to save BM25 cache: {e}")
            return False
    
    def _load_from_cache(self) -> bool:
        """Load BM25 model from cache if available and valid."""
        try:
            if not self.cache_file.exists():
                return False
            
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache version and document count
            if (cache_data.get('cache_version') != '1.0' or 
                cache_data.get('documents_count') != len(self.documents)):
                logger.info("Cache invalid due to version or document count mismatch")
                return False
            
            # Restore BM25 model
            self.tokenized_corpus = cache_data['tokenized_corpus']
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            
            # Restore BM25 state
            self.bm25.idf = cache_data['bm25_idf']
            self.bm25.doc_freqs = cache_data['bm25_doc_freqs']
            self.bm25.doc_len = cache_data['bm25_doc_len']
            self.bm25.avgdl = cache_data['bm25_avgdl']
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load BM25 cache: {e}")
            # Clean up corrupted cache file
            try:
                if self.cache_file.exists():
                    self.cache_file.unlink()
            except:
                pass
            return False
    
    def clear_cache(self):
        """Clear the BM25 cache for this retriever."""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                logger.info(f"Cleared BM25 cache: {self.cache_file}")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
    
    @classmethod
    def clear_all_cache(cls, cache_dir: str = None):
        """Clear all BM25 cache files."""
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), 'rag_bm25_cache')
        
        cache_path = Path(cache_dir)
        if cache_path.exists():
            try:
                for cache_file in cache_path.glob('bm25_*.pkl'):
                    cache_file.unlink()
                logger.info(f"Cleared all BM25 cache files in {cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to clear all cache: {e}")
        
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Deutsche Textvorverarbeitung mit Stemming und Compound-Handling.
        """
        # Normalisiere Text
        text = text.lower()
        
        # Entferne Satzzeichen, behalte Wort-Grenzen
        text = re.sub(r'[^\w\s-]', ' ', text)
        
        # Tokenisiere
        tokens = text.split()
        
        # Use shared stopwords from constants
        # Filtere und stemme
        processed_tokens = []
        for token in tokens:
            if len(token) >= BM25_MIN_TOKEN_LENGTH and token not in COMBINED_STOPWORDS:
                # Stemming für bessere Matching
                stemmed = self.stemmer.stem(token)
                processed_tokens.append(stemmed)
                
                # Compound word handling - füge auch Original hinzu für exakte Matches
                if len(token) > 6:  # Wahrscheinlich compound word
                    processed_tokens.append(token)
        
        return processed_tokens
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """BM25-basierte Keyword-Suche."""
        query_tokens = self._preprocess_text(query)
        
        if not query_tokens:
            return []
        
        # BM25 Scoring
        scores = self.bm25.get_scores(query_tokens)
        
        # Top-K Ergebnisse mit Scores (BM25 kann negative Scores haben)
        results = []
        for idx, score in enumerate(scores):
            result = self.documents[idx].copy()
            result['bm25_score'] = float(score)
            # Normalisiere Score für distance (immer positive Werte)
            result['distance'] = 1.0 / (1.0 + abs(score)) if score < 0 else 1.0 / (1.0 + score)
            results.append(result)
        
        # Sortiere nach Score (höher = besser)
        results.sort(key=lambda x: x['bm25_score'], reverse=True)
        
        # Filtere nur die relevantesten Ergebnisse (mindestens das beste Ergebnis immer zurückgeben)
        if results and top_k > 0:
            # Für kleine Korpora: gib zumindest das beste Ergebnis zurück
            if len(results) <= 3:
                return results[:top_k]
            else:
                # Für größere Korpora: filtere sehr negative Scores aus
                filtered_results = [r for r in results if r['bm25_score'] > -1.0]
                return filtered_results[:top_k] if filtered_results else results[:1]
        
        return results[:top_k]


class EnhancedHybridRetriever:
    """
    Intelligenter Hybrid-Retriever mit adaptiver Gewichtung.
    """
    
    def __init__(self, semantic_retriever, vector_store, documents: List[Dict[str, Any]], 
                 cache_dir: str = None, enable_cache: bool = True):
        self.semantic_retriever = semantic_retriever
        self.vector_store = vector_store
        self.query_analyzer = QueryAnalyzer()
        self.bm25_retriever = GermanBM25Retriever(documents, cache_dir=cache_dir, enable_cache=enable_cache)
        
    def hybrid_retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Führt Hybrid-Retrieval mit adaptiver Gewichtung durch.
        """
        # 1. Query-Analyse für adaptive Gewichtung
        query_analysis = self.query_analyzer.analyze_query_type(query)
        semantic_weight = query_analysis['semantic_weight']
        keyword_weight = query_analysis['keyword_weight']
        
        # 2. Beide Retrieval-Methoden ausführen (mehr Kandidaten holen)
        semantic_results = self.semantic_retriever.retrieve(query, top_k * 3)
        keyword_results = self.bm25_retriever.retrieve(query, top_k * 3)
        
        # 3. Ergebnisse zusammenführen und neu bewerten
        final_results = self._merge_and_rerank(
            semantic_results, keyword_results, 
            semantic_weight, keyword_weight, top_k
        )
        
        # 4. Erweitere Metadaten
        for result in final_results:
            result['retrieval_method'] = 'hybrid'
            result['semantic_weight'] = semantic_weight
            result['keyword_weight'] = keyword_weight
            result['query_type'] = query_analysis['query_type']
        
        return final_results
    
    def _merge_and_rerank(self, semantic_results: List[Dict], keyword_results: List[Dict],
                         semantic_weight: float, keyword_weight: float, top_k: int) -> List[Dict]:
        """Führt intelligente Ergebnis-Zusammenführung durch."""
        
        # Create unified candidate pool mit chunk_id als key
        candidates = {}
        
        # Semantic results verarbeiten
        for i, result in enumerate(semantic_results):
            chunk_id = result['metadata']['chunk_id']
            # Convert distance to score, ensuring non-negative values
            semantic_score = max(0.0, 1.0 - result['distance'])  # Clamp to [0, 1]
            
            candidates[chunk_id] = {
                **result,
                'semantic_score': semantic_score,
                'semantic_rank': i + 1,
                'bm25_score': 0.0,
                'bm25_rank': float('inf')
            }
        
        # BM25 results hinzufügen/aktualisieren
        for i, result in enumerate(keyword_results):
            chunk_id = result['metadata']['chunk_id']
            bm25_score = result['bm25_score']
            
            if chunk_id in candidates:
                # Update existing candidate
                candidates[chunk_id]['bm25_score'] = bm25_score
                candidates[chunk_id]['bm25_rank'] = i + 1
            else:
                # Add new candidate (only from BM25)
                candidates[chunk_id] = {
                    **result,
                    'semantic_score': 0.0,
                    'semantic_rank': float('inf'),
                    'bm25_score': bm25_score,
                    'bm25_rank': i + 1
                }
        
        # Compute hybrid scores
        final_candidates = []
        
        # Get all BM25 scores to normalize properly
        all_bm25_scores = [c['bm25_score'] for c in candidates.values() if c['bm25_score'] != 0]
        min_bm25 = min(all_bm25_scores) if all_bm25_scores else 0
        max_bm25 = max(all_bm25_scores) if all_bm25_scores else 1
        
        for chunk_id, candidate in candidates.items():
            # Normalisiere Scores (0-1 range)
            norm_semantic = candidate['semantic_score']
            
            # Normalize BM25 scores to 0-1 range, handling negative values
            if max_bm25 != min_bm25:
                norm_bm25 = max(0, (candidate['bm25_score'] - min_bm25) / (max_bm25 - min_bm25))
            else:
                norm_bm25 = 0.5 if candidate['bm25_score'] != 0 else 0.0
            
            # Hybrid score mit adaptiven Gewichtungen
            hybrid_score = (semantic_weight * norm_semantic + 
                          keyword_weight * norm_bm25)
            
            # Bonus für Dokumente die in beiden Top-Results sind
            if (candidate['semantic_rank'] <= top_k and 
                candidate['bm25_rank'] <= top_k):
                hybrid_score *= HYBRID_BOOST_FACTOR
            
            candidate['hybrid_score'] = hybrid_score
            candidate['distance'] = 1.0 - hybrid_score  # For consistency
            final_candidates.append(candidate)
        
        # Sortiere nach Hybrid-Score
        final_candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return final_candidates[:top_k]
    
    def clear_bm25_cache(self):
        """Clear the BM25 cache for this retriever."""
        self.bm25_retriever.clear_cache()
    
    def rebuild_bm25_index(self, documents: List[Dict[str, Any]] = None):
        """Rebuild BM25 index with new documents, clearing cache."""
        if documents is not None:
            self.bm25_retriever.clear_cache()
            self.bm25_retriever = GermanBM25Retriever(
                documents, 
                cache_dir=self.bm25_retriever.cache_dir, 
                enable_cache=self.bm25_retriever.enable_cache
            )


if __name__ == '__main__':
    # This section should be moved to a proper test file
    logger.info("Retriever module loaded successfully")
    logger.info("Use this module by importing Retriever, EnhancedHybridRetriever classes")
    logger.info("For testing, run: python -m pytest tests/test_retriever.py")

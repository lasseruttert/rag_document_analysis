from typing import List, Dict, Tuple, Any
import re
from src.embeddings import EmbeddingManager
from src.vectorstore import VectorStoreManager

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
        print(f"Generating embedding for query: '{query}'")
        query_embedding = self.embedding_manager.model.encode(query, convert_to_numpy=True)

        # 2. Ähnlichkeitssuche in der Vektordatenbank
        print(f"Querying vector store for top {top_k} results...")
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
        self.technical_patterns = [
            r'\b[A-Z_]{2,}\b',          # Konstanten/Env vars (API_KEY)
            r'\b\w+\(\)',               # Funktionsaufrufe
            r'\.\w+',                   # Attribute/Methods (.environ)
            r'\w+\.\w+',                # Module.function (os.environ)
            r'\bimport\s+\w+',          # Import statements
            r'\bfrom\s+\w+\s+import',   # From imports
            r'[\w-]+\.ya?ml\b',         # YAML files
            r'[\w-]+\.json\b',          # JSON files
            r'[\w-]+\.py\b',            # Python files
            r'[\w-]+\.js\b',            # JavaScript files
            r'\b\w+_\w+\b',             # Snake_case identifiers
        ]
        self.code_keywords = {
            'python', 'kubernetes', 'docker', 'git', 'api', 'json', 
            'yaml', 'config', 'function', 'class', 'import', 'pip',
            'kubectl', 'pods', 'container', 'npm', 'node', 'java',
            'maven', 'gradle', 'bash', 'shell', 'script', 'command',
            'numpy', 'pandas', 'environment', 'variable', 'environ',
            'compose', 'dockerfile', 'befehl', 'datei'
        }
    
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
        question_patterns = [r'\bwas\b', r'\bwie\b', r'\bwarum\b', r'\bwann\b', r'\?']
        question_score = sum(0.25 for pattern in question_patterns 
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
            semantic_weight = 0.6
            keyword_weight = 0.4
        
        return {
            'semantic_weight': round(semantic_weight, 2),
            'keyword_weight': round(keyword_weight, 2),
            'query_type': 'technical' if technical_total > 0.5 else 
                         'question' if question_score > 0.5 else 'balanced'
        }


class GermanBM25Retriever:
    """BM25-Retriever mit deutscher Sprachoptimierung."""
    
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.stemmer = SnowballStemmer('german')
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Prepare BM25 corpus
        self.tokenized_corpus = [self._preprocess_text(doc['content']) 
                                for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
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
        
        # Deutsche Stoppwörter (erweitert)
        german_stopwords = {
            'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich',
            'des', 'auf', 'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als',
            'auch', 'es', 'an', 'werden', 'aus', 'er', 'hat', 'dass', 'sie',
            'nach', 'wird', 'bei', 'einer', 'um', 'am', 'sind', 'noch', 'wie',
            'einem', 'über', 'einen', 'so', 'zum', 'war', 'haben', 'nur', 'oder',
            'aber', 'vor', 'zur', 'bis', 'mehr', 'durch', 'man', 'sein', 'wurde',
            'wenn', 'können', 'alle', 'würde', 'meine', 'macht', 'kann', 'soll',
            'wir', 'ich', 'dir', 'du', 'ihr', 'uns', 'euch'
        }
        
        # Filtere und stemme
        processed_tokens = []
        for token in tokens:
            if len(token) >= 3 and token not in german_stopwords:
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
    
    def __init__(self, semantic_retriever, vector_store, documents: List[Dict[str, Any]]):
        self.semantic_retriever = semantic_retriever
        self.vector_store = vector_store
        self.query_analyzer = QueryAnalyzer()
        self.bm25_retriever = GermanBM25Retriever(documents)
        
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
                hybrid_score *= 1.2  # 20% boost
            
            candidate['hybrid_score'] = hybrid_score
            candidate['distance'] = 1.0 - hybrid_score  # For consistency
            final_candidates.append(candidate)
        
        # Sortiere nach Hybrid-Score
        final_candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return final_candidates[:top_k]


if __name__ == '__main__':
    # Setup für den Test
    embed_manager = EmbeddingManager()
    vector_store = VectorStoreManager()
    
    # Sicherstellen, dass die Collection existiert und Dokumente enthält
    # (Dieser Teil würde normalerweise in einem Setup-Skript ausgeführt)
    from text_processor import process_documents
    collection_name = "retrieval_test"
    collection = vector_store.create_or_get_collection(collection_name)
    if collection.count() == 0:
        print("Collection is empty. Populating with sample documents...")
        document_chunks = process_documents('data/raw_texts')
        chunks_with_embeddings = embed_manager.generate_embeddings(document_chunks)
        vector_store.add_documents(chunks_with_embeddings)
        print("Population complete.")

    # Retriever initialisieren
    retriever = Retriever(embedding_manager=embed_manager, vector_store_manager=vector_store)

    # Beispiel-Anfragen
    test_queries = [
        "Was ist ein Pod in Kubernetes?",
        "Wie kann ich meine Python-Abhängigkeiten verwalten?",
        "Was sind Best Practices für das Schreiben von Tests in Python?"
    ]

    for user_query in test_queries:
        print(f"\n--- Testing Query: '{user_query}' ---")
        retrieved_docs = retriever.retrieve(user_query, top_k=2)

        print("\nRetrieved Documents:")
        if not retrieved_docs:
            print("No documents found.")
        else:
            for doc in retrieved_docs:
                print(f"  - ID: {doc['id']}")
                print(f"    Distance: {doc['distance']:.4f}")
                print(f"    Content: {doc['content'][:150]}...")
        print("-" * 20)

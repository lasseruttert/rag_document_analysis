from src.text_processor import process_documents
from src.embeddings import EmbeddingManager
from src.vectorstore import VectorStoreManager
from src.retriever import Retriever, EnhancedHybridRetriever
from src.llm_handler import LLMHandler
from src.config import get_config, RAGConfig
from typing import List, Dict, Any, Optional

class RAGPipeline:
    def __init__(self,
                 config: Optional[RAGConfig] = None,
                 text_data_path: Optional[str] = None,
                 vector_db_path: Optional[str] = None,
                 embedding_model_name: Optional[str] = None,
                 llm_model_name: Optional[str] = None):
        """
        Initialisiert die RAG-Pipeline mit allen notwendigen Komponenten.

        Args:
            config: RAGConfig instance (uses global config if None)
            text_data_path: Override for text data path (optional)
            vector_db_path: Override for vector database path (optional)
            embedding_model_name: Override for embedding model (optional)
            llm_model_name: Override for LLM model (optional)
        """
        # Load configuration
        self.config = config or get_config()
        
        # Use provided parameters or fall back to config
        self.text_data_path = text_data_path or self.config.file_processing.input_directory
        self.vector_db_path = vector_db_path or self.config.vector_database.storage_path
        embedding_model = embedding_model_name or self.config.models.embedding_model
        llm_model = llm_model_name or self.config.models.llm_model

        print("Initializing EmbeddingManager...")
        self.embedding_manager = EmbeddingManager(
            model_name=embedding_model,
            batch_size=self.config.vector_database.embedding_batch_size
        )

        print("Initializing VectorStoreManager...")
        self.vector_store_manager = VectorStoreManager(storage_path=self.vector_db_path)
        self.collection = self.vector_store_manager.create_or_get_collection(
            self.config.vector_database.default_collection
        )

        print("Initializing Retriever...")
        self.retriever = Retriever(self.embedding_manager, self.vector_store_manager)

        print("Initializing LLMHandler...")
        self.llm_handler = LLMHandler(
            model_name=llm_model,
            config=self.config.generation
        )
        
        # Enhanced Hybrid Retriever wird bei Bedarf initialisiert
        self.enhanced_retriever = None

    def ingest_documents(self):
        """
        Verarbeitet Dokumente, generiert Embeddings und speichert sie im Vektor-Store.
        """
        print(f"Ingesting documents from {self.text_data_path}...")
        document_chunks = process_documents(
            directory_path=self.text_data_path,
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap
        )
        if not document_chunks:
            print("No documents found to ingest.")
            return

        chunks_with_embeddings = self.embedding_manager.generate_embeddings(document_chunks)
        self.vector_store_manager.add_documents(
            chunks_with_embeddings,
            batch_size=self.config.vector_database.batch_size
        )
        print("Document ingestion complete.")

    def answer_query(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Beantwortet eine Benutzeranfrage unter Verwendung der RAG-Pipeline.

        Args:
            query: Die Suchanfrage des Benutzers.
            top_k: Die Anzahl der relevantesten Dokumente (uses config default if None).

        Returns:
            Die vom LLM generierte Antwort.
        """
        top_k = top_k or self.config.retrieval.default_top_k
        print(f"Processing query: '{query}' (top_k={top_k})")
        
        # 1. Retrieval
        retrieved_chunks = self.retriever.retrieve(query, top_k=top_k)
        
        if not retrieved_chunks:
            return self.config.generation.no_context_response

        # 2. Generierung
        answer = self.llm_handler.generate_answer(query, retrieved_chunks)
        return answer
    
    def _setup_enhanced_retriever(self):
        """Setup Enhanced Hybrid Retriever."""
        print("Setting up Enhanced Hybrid Retriever...")
        
        # Hole alle Dokumente für BM25
        collection_data = self.collection.get(include=['documents', 'metadatas'])
        documents = []
        
        if collection_data['documents']:
            for doc, metadata in zip(collection_data['documents'], collection_data['metadatas']):
                documents.append({
                    'content': doc,
                    'metadata': metadata
                })
        
        # Erstelle Enhanced Hybrid Retriever
        self.enhanced_retriever = EnhancedHybridRetriever(
            semantic_retriever=self.retriever,
            vector_store=self.vector_store_manager,
            documents=documents
        )
        print("Enhanced Hybrid Retriever ready.")

    def enhanced_answer_query(self, query: str, top_k: Optional[int] = None) -> str:
        """Enhanced Query Processing mit Hybrid Retrieval."""
        top_k = top_k or self.config.retrieval.default_top_k
        
        if self.collection.count() == 0:
            return self.config.generation.no_context_response
        
        # Setup Enhanced Retriever if not done
        if not self.enhanced_retriever:
            self._setup_enhanced_retriever()
        
        print(f"Processing enhanced query: '{query}' (top_k={top_k})")
        
        # Hybrid Retrieval
        retrieved_chunks = self.enhanced_retriever.hybrid_retrieve(query, top_k)
        
        if not retrieved_chunks:
            return self.config.generation.no_context_response
        
        # Debug info
        if retrieved_chunks and self.config.development.debug_mode:
            query_type = retrieved_chunks[0].get('query_type', 'unknown')
            semantic_weight = retrieved_chunks[0].get('semantic_weight', 0)
            keyword_weight = retrieved_chunks[0].get('keyword_weight', 0)
            print(f"Query classified as: {query_type} (semantic: {semantic_weight:.2f}, keyword: {keyword_weight:.2f})")
        
        # Generiere Antwort mit Enhanced Context
        answer = self.llm_handler.generate_answer(query, retrieved_chunks)
        
        return answer

if __name__ == '__main__':
    # Beispiel für die Verwendung der Pipeline
    pipeline = RAGPipeline()

    # Dokumente ingestieren (nur einmal ausführen, wenn die DB leer ist)
    if pipeline.collection.count() == 0:
        pipeline.ingest_documents()
    else:
        print(f"Vector store already contains {pipeline.collection.count()} documents. Skipping ingestion.")

    # Beispielanfragen für Standard und Enhanced Retrieval
    queries = [
        "Was ist ein Pod in Kubernetes?",
        "kubectl get pods",  # Technical query
        "Nenne Best Practices für Python Code-Formatierung.",
        "Wie kann ich meine Python-Abhängigkeiten verwalten?",
    ]

    print("\n" + "="*60)
    print("STANDARD RETRIEVAL TESTS")
    print("="*60)
    
    for q in queries:
        print(f"\n--- Frage: {q} ---")
        response = pipeline.answer_query(q)
        print(f"Antwort: {response}")
        print("-" * 50)
    
    print("\n" + "="*60)
    print("ENHANCED HYBRID RETRIEVAL TESTS")
    print("="*60)
    
    for q in queries:
        print(f"\n--- Enhanced Frage: {q} ---")
        response = pipeline.enhanced_answer_query(q)
        print(f"Enhanced Antwort: {response}")
        print("-" * 50)

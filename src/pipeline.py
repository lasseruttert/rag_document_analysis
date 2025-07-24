from src.text_processor import process_documents
from src.embeddings import EmbeddingManager
from src.vectorstore import VectorStoreManager
from src.retriever import Retriever
from src.llm_handler import LLMHandler
from typing import List, Dict

class RAGPipeline:
    def __init__(self,
                 text_data_path: str = 'data/raw_texts',
                 vector_db_path: str = 'data/vectordb',
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 llm_model_name: str = 'google/flan-t5-base'):
        """
        Initialisiert die RAG-Pipeline mit allen notwendigen Komponenten.

        Args:
            text_data_path: Pfad zu den Roh-Textdateien.
            vector_db_path: Pfad zum Speichern der ChromaDB-Daten.
            embedding_model_name: Name des Embedding-Modells.
            llm_model_name: Name des LLM-Modells.
        """
        self.text_data_path = text_data_path
        self.vector_db_path = vector_db_path

        print("Initializing EmbeddingManager...")
        self.embedding_manager = EmbeddingManager(model_name=embedding_model_name)

        print("Initializing VectorStoreManager...")
        self.vector_store_manager = VectorStoreManager(storage_path=self.vector_db_path)
        self.collection = self.vector_store_manager.create_or_get_collection("rag_documents")

        print("Initializing Retriever...")
        self.retriever = Retriever(self.embedding_manager, self.vector_store_manager)

        print("Initializing LLMHandler...")
        self.llm_handler = LLMHandler(model_name=llm_model_name)

    def ingest_documents(self):
        """
        Verarbeitet Dokumente, generiert Embeddings und speichert sie im Vektor-Store.
        """
        print(f"Ingesting documents from {self.text_data_path}...")
        document_chunks = process_documents(self.text_data_path)
        if not document_chunks:
            print("No documents found to ingest.")
            return

        chunks_with_embeddings = self.embedding_manager.generate_embeddings(document_chunks)
        self.vector_store_manager.add_documents(chunks_with_embeddings)
        print("Document ingestion complete.")

    def answer_query(self, query: str, top_k: int = 5) -> str:
        """
        Beantwortet eine Benutzeranfrage unter Verwendung der RAG-Pipeline.

        Args:
            query: Die Suchanfrage des Benutzers.
            top_k: Die Anzahl der relevantesten Dokumente, die abgerufen werden sollen.

        Returns:
            Die vom LLM generierte Antwort.
        """
        print(f"Processing query: '{query}'")
        # 1. Retrieval
        retrieved_chunks = self.retriever.retrieve(query, top_k=top_k)
        
        if not retrieved_chunks:
            return "Es konnten keine relevanten Informationen für Ihre Anfrage gefunden werden."

        # 2. Generierung
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

    # Beispielanfragen
    queries = [
        "Was ist ein Pod in Kubernetes?",
        "Nenne Best Practices für Python Code-Formatierung.",
        "Wie kann ich meine Python-Abhängigkeiten verwalten?",
        "Erzähl mir etwas über die Geschichte von KI." # Sollte keine Antwort finden
    ]

    for q in queries:
        print(f"\n--- Frage: {q} ---")
        response = pipeline.answer_query(q)
        print(f"Antwort: {response}")
        print("-" * 50)

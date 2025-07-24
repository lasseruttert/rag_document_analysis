from typing import List, Dict
from src.embeddings import EmbeddingManager
from src.vectorstore import VectorStoreManager

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

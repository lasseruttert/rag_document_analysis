import chromadb
from typing import List, Dict
import os

class VectorStoreManager:
    def __init__(self, storage_path: str = 'data/vectordb'):
        """
        Initialisiert den VectorStoreManager.

        Args:
            storage_path: Pfad zum Speichern der ChromaDB-Daten.
        """
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        
        self.client = chromadb.PersistentClient(path=storage_path)
        self.collection = None

    def create_or_get_collection(self, name: str = "documents") -> chromadb.Collection:
        """
        Erstellt eine neue Collection oder ruft eine bestehende ab.

        Args:
            name: Name der Collection.

        Returns:
            Die ChromaDB Collection.
        """
        self.collection = self.client.get_or_create_collection(name=name)
        print(f"Collection '{name}' loaded. It contains {self.collection.count()} documents.")
        return self.collection

    def add_documents(self, chunks_with_embeddings: List[Dict[str, any]]):
        """
        Fügt Dokumenten-Chunks zu der Collection hinzu.

        Args:
            chunks_with_embeddings: Eine Liste von Chunks mit generierten Embeddings.
        """
        if not self.collection:
            raise ValueError("Collection not initialized. Call create_or_get_collection() first.")

        if not chunks_with_embeddings:
            print("No documents to add.")
            return

        ids = [chunk['metadata']['chunk_id'] for chunk in chunks_with_embeddings]
        documents = [chunk['content'] for chunk in chunks_with_embeddings]
        embeddings = [chunk['embedding'].tolist() for chunk in chunks_with_embeddings]
        metadatas = [chunk['metadata'] for chunk in chunks_with_embeddings]

        print(f"Adding {len(ids)} documents to the collection...")
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Successfully added documents. Collection now contains {self.collection.count()} documents.")

if __name__ == '__main__':
    # Beispiel für die Verwendung
    from src.text_processor import process_documents
    from src.embeddings import EmbeddingManager

    # 1. Beispieldaten laden und verarbeiten
    data_path = 'data/raw_texts'
    print(f"Loading and processing documents from '{data_path}'...")
    document_chunks = process_documents(data_path)

    if document_chunks:
        # 2. Embeddings generieren
        print("\nInitializing EmbeddingManager...")
        embed_manager = EmbeddingManager()
        chunks_with_embeddings = embed_manager.generate_embeddings(document_chunks)

        # 3. VectorStoreManager initialisieren und Dokumente hinzufügen
        print("\nInitializing VectorStoreManager...")
        vector_store = VectorStoreManager()
        collection = vector_store.create_or_get_collection("my_documents")
        vector_store.add_documents(chunks_with_embeddings)

        # 4. Überprüfung
        print(f"\nVerification: Collection '{collection.name}' now has {collection.count()} entries.")
        # Optional: Ein Element abrufen, um es zu überprüfen
        retrieved = collection.get(ids=[chunks_with_embeddings[0]['metadata']['chunk_id']], include=["metadatas", "documents"])
        print("\nSample retrieved item:")
        print(retrieved)

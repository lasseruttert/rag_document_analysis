import chromadb
from typing import List, Dict, Optional
import os
from src.config import get_config, VectorDatabaseConfig

class VectorStoreManager:
    def __init__(self, 
                 storage_path: Optional[str] = None,
                 config: Optional[VectorDatabaseConfig] = None):
        """
        Initialisiert den VectorStoreManager.

        Args:
            storage_path: Pfad zum Speichern der ChromaDB-Daten (uses config if None).
            config: VectorDatabaseConfig instance (uses global config if None).
        """
        # Load configuration
        if config is None:
            full_config = get_config()
            config = full_config.vector_database
        
        self.config = config
        storage_path = storage_path or config.storage_path
        
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        
        self.client = chromadb.PersistentClient(path=storage_path)
        self.collection = None

    def create_or_get_collection(self, name: Optional[str] = None) -> chromadb.Collection:
        """
        Erstellt eine neue Collection oder ruft eine bestehende ab.

        Args:
            name: Name der Collection (uses config default if None).

        Returns:
            Die ChromaDB Collection.
        """
        name = name or self.config.default_collection
        self.collection = self.client.get_or_create_collection(name=name)
        print(f"Collection '{name}' loaded. It contains {self.collection.count()} documents.")
        return self.collection

    def add_documents(self, chunks_with_embeddings: List[Dict[str, any]], batch_size: Optional[int] = None):
        """
        Fügt Dokumenten-Chunks zu der Collection hinzu.

        Args:
            chunks_with_embeddings: Eine Liste von Chunks mit generierten Embeddings.
            batch_size: Batch-Größe für die Verarbeitung (uses config default if None).
        """
        if not self.collection:
            raise ValueError("Collection not initialized. Call create_or_get_collection() first.")

        if not chunks_with_embeddings:
            print("No documents to add.")
            return

        batch_size = batch_size or self.config.batch_size
        total_chunks = len(chunks_with_embeddings)
        
        print(f"Adding {total_chunks} documents to the collection in batches of {batch_size}...")
        
        # Process in batches
        for i in range(0, total_chunks, batch_size):
            batch_end = min(i + batch_size, total_chunks)
            batch_chunks = chunks_with_embeddings[i:batch_end]
            
            ids = [chunk['metadata']['chunk_id'] for chunk in batch_chunks]
            documents = [chunk['content'] for chunk in batch_chunks]
            embeddings = [chunk['embedding'].tolist() for chunk in batch_chunks]
            metadatas = [chunk['metadata'] for chunk in batch_chunks]

            print(f"Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} ({len(batch_chunks)} documents)...")
            
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

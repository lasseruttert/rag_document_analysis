import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np

class EmbeddingManager:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', use_gpu: bool = True):
        """
        Initialisiert den EmbeddingManager.

        Args:
            model_name: Name des Sentence-Transformer-Modells.
            use_gpu: Ob die GPU verwendet werden soll, falls verfügbar.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def generate_embeddings(self, chunks: List[Dict[str, any]], batch_size: int = 32) -> List[Dict[str, any]]:
        """
        Generiert Embeddings für eine Liste von Text-Chunks.

        Args:
            chunks: Eine Liste von Dictionaries, die die Chunks enthalten.
            batch_size: Die Größe der Batches für die Verarbeitung.

        Returns:
            Die Liste der Chunks, angereichert mit den generierten Embeddings.
        """
        if not chunks:
            return []

        # Extrahieren der Inhalte für das Encoding
        contents = [chunk['content'] for chunk in chunks]
        
        print(f"Generating embeddings for {len(contents)} chunks in batches of {batch_size}...")
        
        # Generieren der Embeddings in Batches
        embeddings = self.model.encode(
            contents, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )
        
        # Hinzufügen der Embeddings zu den Chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i]
            
        return chunks

if __name__ == '__main__':
    # Beispiel für die Verwendung
    from src.text_processor import process_documents

    # 1. Beispieldaten laden und verarbeiten
    data_path = 'data/raw_texts'
    print(f"Loading and processing documents from '{data_path}'...")
    document_chunks = process_documents(data_path)

    if not document_chunks:
        print("No documents to process.")
    else:
        # 2. EmbeddingManager initialisieren
        print("\nInitializing EmbeddingManager...")
        embed_manager = EmbeddingManager()

        # 3. Embeddings generieren
        chunks_with_embeddings = embed_manager.generate_embeddings(document_chunks)

        # 4. Ergebnisse überprüfen
        print(f"\nSuccessfully generated embeddings for {len(chunks_with_embeddings)} chunks.")
        if chunks_with_embeddings:
            sample_chunk = chunks_with_embeddings[0]
            print("\nSample chunk with embedding:")
            print(f"  Filename: {sample_chunk['metadata']['filename']}")
            print(f"  Content: {sample_chunk['content'][:100]}...")
            print(f"  Embedding shape: {sample_chunk['embedding'].shape}")

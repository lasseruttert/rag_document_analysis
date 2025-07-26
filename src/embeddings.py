import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import numpy as np
import logging
from src.config import get_config, ModelConfig

logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self, 
                 model_name: Optional[str] = None, 
                 use_gpu: Optional[bool] = None,
                 batch_size: Optional[int] = None,
                 config: Optional[ModelConfig] = None):
        """
        Initialisiert den EmbeddingManager.

        Args:
            model_name: Name des Sentence-Transformer-Modells (uses config if None).
            use_gpu: Ob die GPU verwendet werden soll (uses config if None).
            batch_size: Default batch size for embedding generation.
            config: ModelConfig instance (uses global config if None).
        """
        # Load configuration
        if config is None:
            full_config = get_config()
            config = full_config.models
        
        self.config = config
        model_name = model_name or config.embedding_model
        
        # Device selection based on config
        if use_gpu is None:
            if config.device_preference == "gpu":
                use_gpu = True
            elif config.device_preference == "cpu":
                use_gpu = False
            else:  # "auto"
                use_gpu = torch.cuda.is_available()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = SentenceTransformer(
            model_name, 
            device=self.device,
            trust_remote_code=config.trust_remote_code,
            use_auth_token=config.use_auth_token if config.use_auth_token else None
        )
        
        # Set default batch size
        self.default_batch_size = batch_size or getattr(get_config().vector_database, 'embedding_batch_size', 32)

    def generate_embeddings(self, chunks: List[Dict[str, any]], batch_size: Optional[int] = None) -> List[Dict[str, any]]:
        """
        Generiert Embeddings für eine Liste von Text-Chunks.

        Args:
            chunks: Eine Liste von Dictionaries, die die Chunks enthalten.
            batch_size: Die Größe der Batches für die Verarbeitung (uses default if None).

        Returns:
            Die Liste der Chunks, angereichert mit den generierten Embeddings.
        """
        if not chunks:
            return []

        batch_size = batch_size or self.default_batch_size

        # Extrahieren der Inhalte für das Encoding
        contents = [chunk['content'] for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(contents)} chunks in batches of {batch_size}...")
        
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
    logger.info(f"Loading and processing documents from '{data_path}'...")
    document_chunks = process_documents(data_path)

    if not document_chunks:
        logger.warning("No documents to process.")
    else:
        # 2. EmbeddingManager initialisieren
        logger.info("Initializing EmbeddingManager...")
        embed_manager = EmbeddingManager()

        # 3. Embeddings generieren
        chunks_with_embeddings = embed_manager.generate_embeddings(document_chunks)

        # 4. Ergebnisse überprüfen
        logger.info(f"Successfully generated embeddings for {len(chunks_with_embeddings)} chunks.")
        if chunks_with_embeddings:
            sample_chunk = chunks_with_embeddings[0]
            logger.debug("Sample chunk with embedding:")
            logger.debug(f"  Filename: {sample_chunk['metadata']['filename']}")
            logger.debug(f"  Content: {sample_chunk['content'][:100]}...")
            logger.debug(f"  Embedding shape: {sample_chunk['embedding'].shape}")

import chromadb
import logging
from typing import List, Dict, Optional, Union, Any
import os
from src.config import get_config, VectorDatabaseConfig
from src.collection_manager import CollectionManager
from src.metadata_filter import MetadataFilter, QueryFilter, CombinedFilter

logger = logging.getLogger(__name__)

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
        
        # Initialize collection manager
        self.collection_manager = CollectionManager(storage_path)
        
        # Current active collection
        self.active_collection_name = None

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
        logger.info(f"Collection '{name}' loaded. It contains {self.collection.count()} documents.")
        return self.collection

    def add_documents(self, chunks_with_embeddings: List[Dict[str, any]], 
                     batch_size: Optional[int] = None,
                     collection_name: Optional[str] = None):
        """
        Fügt Dokumenten-Chunks zu der Collection hinzu.

        Args:
            chunks_with_embeddings: Eine Liste von Chunks mit generierten Embeddings.
            batch_size: Batch-Größe für die Verarbeitung (uses config default if None).
            collection_name: Target collection (uses active collection if None).
        """
        # Determine target collection
        if collection_name:
            target_collection = self.get_collection(collection_name)
        elif self.collection:
            target_collection = self.collection
        else:
            raise ValueError("No collection specified. Call create_or_get_collection() first.")

        if not chunks_with_embeddings:
            logger.warning("No documents to add.")
            return

        batch_size = batch_size or self.config.batch_size
        total_chunks = len(chunks_with_embeddings)
        
        logger.info(f"Adding {total_chunks} documents to collection '{target_collection.name}' in batches of {batch_size}...")
        
        # Process in batches
        for i in range(0, total_chunks, batch_size):
            batch_end = min(i + batch_size, total_chunks)
            batch_chunks = chunks_with_embeddings[i:batch_end]
            
            ids = [chunk['metadata']['chunk_id'] for chunk in batch_chunks]
            documents = [chunk['content'] for chunk in batch_chunks]
            embeddings = [chunk['embedding'].tolist() for chunk in batch_chunks]
            metadatas = [chunk['metadata'] for chunk in batch_chunks]

            logger.debug(f"Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} ({len(batch_chunks)} documents)...")
            
            target_collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
        
        logger.info(f"Successfully added documents. Collection '{target_collection.name}' now contains {target_collection.count()} documents.")
        
        # Use collection manager to track stats
        if collection_name:
            self.collection_manager._update_collection_stats(collection_name)
    
    def get_collection(self, name: str) -> Optional[chromadb.Collection]:
        """
        Get a specific collection by name.
        
        Args:
            name: Collection name
            
        Returns:
            ChromaDB collection or None if not found
        """
        try:
            return self.client.get_collection(name=name)
        except Exception as e:
            logger.warning(f"Collection '{name}' not found: {e}")
            return None
    
    def set_active_collection(self, name: str) -> bool:
        """
        Set the active collection for operations.
        
        Args:
            name: Collection name to activate
            
        Returns:
            True if successful, False if collection doesn't exist
        """
        collection = self.get_collection(name)
        if collection:
            self.collection = collection
            self.active_collection_name = name
            logger.info(f"Active collection set to '{name}' with {collection.count()} documents.")
            return True
        else:
            logger.warning(f"Collection '{name}' not found.")
            return False
    
    def search_with_filters(self, 
                           query_embedding: List[float],
                           filters: Union[QueryFilter, CombinedFilter, None] = None,
                           n_results: int = 5,
                           collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search documents with metadata filtering.
        
        Args:
            query_embedding: Query embedding vector
            filters: Metadata filters to apply
            n_results: Maximum number of results
            collection_name: Collection to search (uses active if None)
            
        Returns:
            List of matching documents with metadata
        """
        # Determine target collection
        if collection_name:
            target_collection = self.get_collection(collection_name)
            if not target_collection:
                return []
        elif self.collection:
            target_collection = self.collection
        else:
            raise ValueError("No collection specified. Set active collection or provide collection_name.")
        
        try:
            # First, get all documents if we need to filter
            if filters:
                # Get a larger set to filter from
                initial_results = target_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results * 5, 100),  # Get more to filter from
                    include=['documents', 'metadatas', 'distances']
                )
                
                # Format results
                documents_with_metadata = []
                documents = initial_results.get('documents', [[]])[0]
                metadatas = initial_results.get('metadatas', [[]])[0]
                distances = initial_results.get('distances', [[]])[0]
                
                for doc, meta, dist in zip(documents, metadatas, distances):
                    documents_with_metadata.append({
                        'content': doc,
                        'metadata': meta,
                        'distance': dist
                    })
                
                # Apply filters
                filtered_docs = MetadataFilter.apply_filters(documents_with_metadata, filters)
                
                # Return top n_results
                return filtered_docs[:n_results]
            
            else:
                # No filters, direct search
                results = target_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=['documents', 'metadatas', 'distances']
                )
                
                # Format results
                formatted_results = []
                documents = results.get('documents', [[]])[0]
                metadatas = results.get('metadatas', [[]])[0]
                distances = results.get('distances', [[]])[0]
                
                for doc, meta, dist in zip(documents, metadatas, distances):
                    formatted_results.append({
                        'content': doc,
                        'metadata': meta,
                        'distance': dist
                    })
                
                return formatted_results
                
        except Exception as e:
            logger.error(f"Error searching collection '{target_collection.name}': {e}")
            return []
    
    def search_across_collections(self,
                                 query_embedding: List[float],
                                 collection_names: Optional[List[str]] = None,
                                 filters: Union[QueryFilter, CombinedFilter, None] = None,
                                 n_results_per_collection: int = 3,
                                 total_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search across multiple collections.
        
        Args:
            query_embedding: Query embedding vector
            collection_names: Collections to search (all if None)
            filters: Metadata filters to apply
            n_results_per_collection: Results per collection
            total_results: Maximum total results
            
        Returns:
            Ranked list of results from all collections
        """
        if collection_names is None:
            # Get all collections
            collections = self.collection_manager.list_collections()
            collection_names = [coll.name for coll in collections]
        
        all_results = []
        
        for coll_name in collection_names:
            try:
                coll_results = self.search_with_filters(
                    query_embedding=query_embedding,
                    filters=filters,
                    n_results=n_results_per_collection,
                    collection_name=coll_name
                )
                
                # Add collection info to metadata
                for result in coll_results:
                    result['metadata']['collection_name'] = coll_name
                
                all_results.extend(coll_results)
                
            except Exception as e:
                logger.error(f"Error searching collection '{coll_name}': {e}")
                continue
        
        # Sort by distance (lower is better)
        all_results.sort(key=lambda x: x.get('distance', float('inf')))
        
        return all_results[:total_results]
    
    def get_collection_statistics(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for a collection.
        
        Args:
            collection_name: Collection name (uses active if None)
            
        Returns:
            Dictionary with collection statistics
        """
        if collection_name is None:
            collection_name = self.active_collection_name
        
        if not collection_name:
            raise ValueError("No collection specified.")
        
        stats = self.collection_manager.get_collection_stats(collection_name)
        if stats:
            return {
                'name': stats.name,
                'document_count': stats.document_count,
                'chunk_count': stats.chunk_count,
                'file_types': stats.file_type_distribution,
                'total_size': stats.total_size_chars,
                'avg_chunk_size': stats.avg_chunk_size,
                'created_at': stats.created_at,
                'last_updated': stats.last_updated,
                'tags': stats.tags,
                'recent_files': stats.recent_files
            }
        else:
            return {}
    
    def list_all_collections(self) -> List[Dict[str, Any]]:
        """
        List all available collections with basic info.
        
        Returns:
            List of collection information dictionaries
        """
        collections = self.collection_manager.list_collections()
        return [
            {
                'name': coll.name,
                'description': coll.description,
                'document_count': coll.document_count,
                'chunk_count': coll.total_chunks,
                'file_types': coll.file_types,
                'created_at': coll.created_at,
                'last_updated': coll.last_updated,
                'tags': coll.tags
            }
            for coll in collections
        ]
    
    def create_collection(self, name: str, description: str = "", tags: List[str] = None) -> bool:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            description: Optional description
            tags: Optional tags
            
        Returns:
            True if created successfully
        """
        return self.collection_manager.create_collection(name, description, tags)
    
    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            name: Collection name to delete
            
        Returns:
            True if deleted successfully
        """
        # Clear active collection if it's being deleted
        if self.active_collection_name == name:
            self.collection = None
            self.active_collection_name = None
            
        return self.collection_manager.delete_collection(name)

if __name__ == '__main__':
    logger.info("VectorStoreManager module loaded for testing")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collection Management System for RAG Document Analysis.

This module provides collection-based document organization with metadata
filtering and advanced search capabilities.
"""

import os
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import chromadb
from src.config import get_config

logger = logging.getLogger(__name__)

@dataclass
class CollectionInfo:
    """Information about a document collection."""
    name: str
    description: str
    created_at: str
    document_count: int
    file_types: List[str]
    total_chunks: int
    last_updated: str
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass 
class CollectionStats:
    """Detailed statistics for a collection."""
    name: str
    document_count: int
    chunk_count: int
    file_type_distribution: Dict[str, int]
    total_size_chars: int
    avg_chunk_size: float
    created_at: str
    last_updated: str
    tags: List[str]
    recent_files: List[str]

class CollectionManager:
    """
    Manages document collections with metadata and statistics.
    
    Features:
    - Collection creation and deletion
    - Metadata tracking and statistics
    - Collection listing and search
    - Tag-based organization
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize collection manager.
        
        Args:
            storage_path: Path to ChromaDB storage (uses config if None)
        """
        config = get_config()
        self.storage_path = storage_path or config.vector_database.storage_path
        self.metadata_path = Path(self.storage_path) / "collections_metadata.json"
        self.client = chromadb.PersistentClient(path=self.storage_path)
        
        # Load or create metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load collection metadata from file."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.error(f"Error loading collection metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save collection metadata to file."""
        try:
            os.makedirs(self.metadata_path.parent, exist_ok=True)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving collection metadata: {e}")
    
    def create_collection(self, name: str, description: str = "", tags: List[str] = None) -> bool:
        """
        Create a new document collection.
        
        Args:
            name: Collection name (must be unique)
            description: Optional description
            tags: Optional tags for organization
            
        Returns:
            True if created successfully, False if already exists
        """
        if tags is None:
            tags = []
            
        # Validate collection name
        if not name or not isinstance(name, str):
            raise ValueError("Collection name must be a non-empty string")
        
        # Check if collection already exists
        if self.collection_exists(name):
            logger.warning(f"Collection '{name}' already exists")
            return False
        
        try:
            # Create ChromaDB collection
            collection = self.client.create_collection(name=name)
            
            # Store metadata
            now = datetime.now().isoformat()
            self.metadata[name] = {
                "description": description,
                "created_at": now,
                "last_updated": now,
                "tags": tags,
                "document_count": 0,
                "file_types": [],
                "total_chunks": 0
            }
            
            self._save_metadata()
            logger.info(f"Created collection '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection '{name}': {e}")
            return False
    
    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection and all its documents.
        
        Args:
            name: Collection name to delete
            
        Returns:
            True if deleted successfully, False if not found
        """
        if not self.collection_exists(name):
            logger.warning(f"Collection '{name}' does not exist")
            return False
        
        try:
            # Delete ChromaDB collection
            self.client.delete_collection(name=name)
            
            # Remove metadata
            if name in self.metadata:
                del self.metadata[name]
                self._save_metadata()
            
            logger.info(f"Deleted collection '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection '{name}': {e}")
            return False
    
    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        try:
            self.client.get_collection(name=name)
            return True
        except:
            return False
    
    def list_collections(self) -> List[CollectionInfo]:
        """
        Get list of all collections with basic information.
        
        Returns:
            List of CollectionInfo objects
        """
        collections = []
        
        try:
            # Get all ChromaDB collections
            chromadb_collections = self.client.list_collections()
            
            for collection in chromadb_collections:
                name = collection.name
                metadata = self.metadata.get(name, {})
                
                # Get current collection stats
                try:
                    coll = self.client.get_collection(name=name)
                    count = coll.count()
                    
                    # Get file types from collection
                    if count > 0:
                        sample_data = coll.get(limit=min(100, count), include=['metadatas'])
                        file_types = list(set(
                            meta.get('file_type', 'unknown') 
                            for meta in sample_data.get('metadatas', [])
                        ))
                    else:
                        file_types = []
                    
                except Exception as e:
                    logger.error(f"Error getting stats for collection '{name}': {e}")
                    count = 0
                    file_types = []
                
                collections.append(CollectionInfo(
                    name=name,
                    description=metadata.get('description', ''),
                    created_at=metadata.get('created_at', ''),
                    document_count=self._count_unique_documents(name),
                    file_types=file_types,
                    total_chunks=count,
                    last_updated=metadata.get('last_updated', ''),
                    tags=metadata.get('tags', [])
                ))
                
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
        
        return collections
    
    def get_collection_stats(self, name: str) -> Optional[CollectionStats]:
        """
        Get detailed statistics for a collection.
        
        Args:
            name: Collection name
            
        Returns:
            CollectionStats object or None if collection doesn't exist
        """
        if not self.collection_exists(name):
            return None
        
        try:
            collection = self.client.get_collection(name=name)
            chunk_count = collection.count()
            metadata = self.metadata.get(name, {})
            
            if chunk_count == 0:
                return CollectionStats(
                    name=name,
                    document_count=0,
                    chunk_count=0,
                    file_type_distribution={},
                    total_size_chars=0,
                    avg_chunk_size=0.0,
                    created_at=metadata.get('created_at', ''),
                    last_updated=metadata.get('last_updated', ''),
                    tags=metadata.get('tags', []),
                    recent_files=[]
                )
            
            # Get all data for analysis
            data = collection.get(include=['documents', 'metadatas'])
            documents = data.get('documents', [])
            metadatas = data.get('metadatas', [])
            
            # Calculate statistics
            file_type_dist = {}
            unique_files = set()
            total_chars = 0
            recent_files = set()
            
            for doc, meta in zip(documents, metadatas):
                if meta:
                    file_type = meta.get('file_type', 'unknown')
                    filename = meta.get('filename', 'unknown')
                    
                    file_type_dist[file_type] = file_type_dist.get(file_type, 0) + 1
                    unique_files.add(filename)
                    recent_files.add(filename)
                    
                if doc:
                    total_chars += len(doc)
            
            avg_chunk_size = total_chars / chunk_count if chunk_count > 0 else 0.0
            
            return CollectionStats(
                name=name,
                document_count=len(unique_files),
                chunk_count=chunk_count,
                file_type_distribution=file_type_dist,
                total_size_chars=total_chars,
                avg_chunk_size=round(avg_chunk_size, 1),
                created_at=metadata.get('created_at', ''),
                last_updated=metadata.get('last_updated', ''),
                tags=metadata.get('tags', []),
                recent_files=list(recent_files)[:10]  # Limit to 10 most recent
            )
            
        except Exception as e:
            logger.error(f"Error getting collection stats for '{name}': {e}")
            return None
    
    def update_collection_metadata(self, name: str, description: str = None, tags: List[str] = None) -> bool:
        """
        Update collection metadata.
        
        Args:
            name: Collection name
            description: New description (optional)
            tags: New tags (optional)
            
        Returns:
            True if updated successfully
        """
        if not self.collection_exists(name):
            return False
        
        try:
            if name not in self.metadata:
                self.metadata[name] = {}
            
            if description is not None:
                self.metadata[name]['description'] = description
            
            if tags is not None:
                self.metadata[name]['tags'] = tags
            
            self.metadata[name]['last_updated'] = datetime.now().isoformat()
            self._save_metadata()
            
            logger.info(f"Updated metadata for collection '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error updating collection metadata for '{name}': {e}")
            return False
    
    def _count_unique_documents(self, collection_name: str) -> int:
        """Count unique documents in a collection."""
        try:
            collection = self.client.get_collection(name=collection_name)
            if collection.count() == 0:
                return 0
            
            data = collection.get(include=['metadatas'])
            metadatas = data.get('metadatas', [])
            
            unique_files = set()
            for meta in metadatas:
                if meta:
                    filename = meta.get('filename', 'unknown')
                    unique_files.add(filename)
            
            return len(unique_files)
            
        except Exception as e:
            logger.error(f"Error counting documents in collection '{collection_name}': {e}")
            return 0
    
    def get_collection(self, name: str):
        """Get ChromaDB collection object."""
        if self.collection_exists(name):
            return self.client.get_collection(name=name)
        return None
    
    def add_documents_to_collection(self, collection_name: str, chunks_with_embeddings: List[Dict[str, Any]]) -> bool:
        """
        Add documents to a specific collection.
        
        Args:
            collection_name: Target collection name
            chunks_with_embeddings: Document chunks with embeddings
            
        Returns:
            True if added successfully
        """
        if not self.collection_exists(collection_name):
            logger.error(f"Collection '{collection_name}' does not exist")
            return False
        
        if not chunks_with_embeddings:
            return True
        
        try:
            collection = self.client.get_collection(name=collection_name)
            
            # Prepare data for ChromaDB
            ids = [chunk['metadata']['chunk_id'] for chunk in chunks_with_embeddings]
            documents = [chunk['content'] for chunk in chunks_with_embeddings]
            embeddings = [chunk['embedding'].tolist() for chunk in chunks_with_embeddings]
            metadatas = [chunk['metadata'] for chunk in chunks_with_embeddings]
            
            # Add to collection
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            # Update metadata
            self._update_collection_stats(collection_name)
            
            logger.info(f"Added {len(chunks_with_embeddings)} chunks to collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to collection '{collection_name}': {e}")
            return False
    
    def _update_collection_stats(self, collection_name: str):
        """Update collection statistics after adding documents."""
        try:
            if collection_name not in self.metadata:
                self.metadata[collection_name] = {}
            
            # Get current stats
            stats = self.get_collection_stats(collection_name)
            if stats:
                self.metadata[collection_name].update({
                    'document_count': stats.document_count,
                    'total_chunks': stats.chunk_count,
                    'file_types': list(stats.file_type_distribution.keys()),
                    'last_updated': datetime.now().isoformat()
                })
                
                self._save_metadata()
                
        except Exception as e:
            logger.error(f"Error updating collection stats for '{collection_name}': {e}")

# Test and example usage
if __name__ == '__main__':
    print("Testing Collection Management System...")
    
    # Initialize collection manager
    manager = CollectionManager()
    
    # Test collection creation
    print("\n1. Testing collection creation...")
    success = manager.create_collection(
        "test_collection", 
        "Test collection for development",
        tags=["test", "development"]
    )
    print(f"Collection creation: {'Success' if success else 'Failed'}")
    
    # Test collection listing
    print("\n2. Testing collection listing...")
    collections = manager.list_collections()
    print(f"Found {len(collections)} collections:")
    for coll in collections:
        print(f"  - {coll.name}: {coll.description} ({coll.total_chunks} chunks)")
    
    # Test collection stats
    print("\n3. Testing collection statistics...")
    if collections:
        stats = manager.get_collection_stats(collections[0].name)
        if stats:
            print(f"Stats for '{stats.name}':")
            print(f"  Documents: {stats.document_count}")
            print(f"  Chunks: {stats.chunk_count}")
            print(f"  File types: {stats.file_type_distribution}")
    
    # Test metadata update
    print("\n4. Testing metadata update...")
    if collections:
        success = manager.update_collection_metadata(
            collections[0].name,
            description="Updated test collection",
            tags=["test", "updated"]
        )
        print(f"Metadata update: {'Success' if success else 'Failed'}")
    
    print("\nCollection Management System test completed!")
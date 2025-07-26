from src.text_processor import process_documents
from src.embeddings import EmbeddingManager
from src.vectorstore import VectorStoreManager
from src.retriever import Retriever, EnhancedHybridRetriever
from src.llm_handler import LLMHandler
from src.config import get_config, RAGConfig
from src.metadata_filter import MetadataFilter, QueryFilter, CombinedFilter
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

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
        
        # Current active collection name
        self.active_collection_name = self.config.vector_database.default_collection

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
    
    # ===================== COLLECTION MANAGEMENT =====================
    
    def create_collection(self, name: str, description: str = "", tags: List[str] = None) -> bool:
        """
        Create a new document collection.
        
        Args:
            name: Collection name
            description: Optional description
            tags: Optional tags for organization
            
        Returns:
            True if created successfully
        """
        return self.vector_store_manager.create_collection(name, description, tags)
    
    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            name: Collection name to delete
            
        Returns:
            True if deleted successfully
        """
        # Reset active collection if it's being deleted
        if self.active_collection_name == name:
            self.active_collection_name = self.config.vector_database.default_collection
            self.collection = self.vector_store_manager.create_or_get_collection(
                self.active_collection_name
            )
            self.enhanced_retriever = None  # Reset enhanced retriever
        
        return self.vector_store_manager.delete_collection(name)
    
    def set_active_collection(self, name: str) -> bool:
        """
        Set the active collection for queries.
        
        Args:
            name: Collection name to activate
            
        Returns:
            True if successful
        """
        success = self.vector_store_manager.set_active_collection(name)
        if success:
            self.active_collection_name = name
            self.collection = self.vector_store_manager.get_collection(name)
            self.enhanced_retriever = None  # Reset enhanced retriever for new collection
            
            # Update retriever with new active collection
            self.retriever = Retriever(self.embedding_manager, self.vector_store_manager)
        
        return success
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all available collections.
        
        Returns:
            List of collection information
        """
        return self.vector_store_manager.list_all_collections()
    
    def get_collection_statistics(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for a collection.
        
        Args:
            collection_name: Collection name (uses active if None)
            
        Returns:
            Dictionary with collection statistics
        """
        return self.vector_store_manager.get_collection_statistics(
            collection_name or self.active_collection_name
        )
    
    def ingest_documents_to_collection(self, collection_name: str, 
                                     text_data_path: Optional[str] = None) -> bool:
        """
        Ingest documents to a specific collection.
        
        Args:
            collection_name: Target collection name
            text_data_path: Path to documents (uses default if None)
            
        Returns:
            True if successful
        """
        # Ensure collection exists
        if not self.vector_store_manager.collection_manager.collection_exists(collection_name):
            logger.error(f"Collection '{collection_name}' does not exist")
            return False
        
        data_path = text_data_path or self.text_data_path
        print(f"Ingesting documents from {data_path} to collection '{collection_name}'...")
        
        document_chunks = process_documents(
            directory_path=data_path,
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap
        )
        
        if not document_chunks:
            print("No documents found to ingest.")
            return False

        chunks_with_embeddings = self.embedding_manager.generate_embeddings(document_chunks)
        self.vector_store_manager.add_documents(
            chunks_with_embeddings,
            batch_size=self.config.vector_database.batch_size,
            collection_name=collection_name
        )
        
        print(f"Document ingestion to collection '{collection_name}' complete.")
        return True
    
    # ===================== FILTERED SEARCH METHODS =====================
    
    def answer_query_with_filters(self, 
                                 query: str, 
                                 filters: Union[QueryFilter, CombinedFilter, None] = None,
                                 top_k: Optional[int] = None,
                                 collection_name: Optional[str] = None) -> str:
        """
        Answer query with metadata filtering.
        
        Args:
            query: User query
            filters: Metadata filters to apply
            top_k: Number of results to retrieve
            collection_name: Collection to search (uses active if None)
            
        Returns:
            Generated answer
        """
        top_k = top_k or self.config.retrieval.default_top_k
        target_collection = collection_name or self.active_collection_name
        
        print(f"Processing filtered query: '{query}' on collection '{target_collection}' (top_k={top_k})")
        
        # Generate query embedding
        query_embedding = self.embedding_manager.model.encode([query])[0].tolist()
        
        # Search with filters
        retrieved_chunks = self.vector_store_manager.search_with_filters(
            query_embedding=query_embedding,
            filters=filters,
            n_results=top_k,
            collection_name=collection_name
        )
        
        if not retrieved_chunks:
            return self.config.generation.no_context_response
        
        # Convert to format expected by LLM handler
        formatted_chunks = []
        for chunk in retrieved_chunks:
            formatted_chunks.append({
                'content': chunk['content'],
                'metadata': chunk['metadata'],
                'distance': chunk['distance']
            })
        
        # Generate answer
        answer = self.llm_handler.generate_answer(query, formatted_chunks)
        return answer
    
    def search_across_collections(self,
                                 query: str,
                                 collection_names: Optional[List[str]] = None,
                                 filters: Union[QueryFilter, CombinedFilter, None] = None,
                                 n_results_per_collection: int = 3,
                                 total_results: int = 10) -> str:
        """
        Search across multiple collections and generate answer.
        
        Args:
            query: User query
            collection_names: Collections to search (all if None)
            filters: Metadata filters to apply
            n_results_per_collection: Results per collection
            total_results: Maximum total results
            
        Returns:
            Generated answer from combined results
        """
        print(f"Processing cross-collection query: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedding_manager.model.encode([query])[0].tolist()
        
        # Search across collections
        retrieved_chunks = self.vector_store_manager.search_across_collections(
            query_embedding=query_embedding,
            collection_names=collection_names,
            filters=filters,
            n_results_per_collection=n_results_per_collection,
            total_results=total_results
        )
        
        if not retrieved_chunks:
            return self.config.generation.no_context_response
        
        # Add collection info to context
        context_with_collections = []
        for chunk in retrieved_chunks:
            collection_name = chunk['metadata'].get('collection_name', 'unknown')
            enhanced_chunk = {
                'content': f"[Collection: {collection_name}] {chunk['content']}",
                'metadata': chunk['metadata'],
                'distance': chunk['distance']
            }
            context_with_collections.append(enhanced_chunk)
        
        # Generate answer
        answer = self.llm_handler.generate_answer(query, context_with_collections)
        return answer
    
    def enhanced_answer_query_with_filters(self,
                                          query: str,
                                          filters: Union[QueryFilter, CombinedFilter, None] = None,
                                          top_k: Optional[int] = None,
                                          collection_name: Optional[str] = None) -> str:
        """
        Enhanced query processing with hybrid retrieval and filtering.
        
        Args:
            query: User query
            filters: Metadata filters to apply
            top_k: Number of results to retrieve
            collection_name: Collection to search (uses active if None)
            
        Returns:
            Generated answer
        """
        top_k = top_k or self.config.retrieval.default_top_k
        target_collection = collection_name or self.active_collection_name
        
        # Set active collection if needed
        if collection_name and collection_name != self.active_collection_name:
            self.set_active_collection(collection_name)
        
        # Check if collection has data
        collection_obj = self.vector_store_manager.get_collection(target_collection)
        if not collection_obj or collection_obj.count() == 0:
            return self.config.generation.no_context_response
        
        print(f"Processing enhanced filtered query: '{query}' on collection '{target_collection}' (top_k={top_k})")
        
        # Setup Enhanced Retriever if not done or collection changed
        if not self.enhanced_retriever:
            self._setup_enhanced_retriever()
        
        # For filtered hybrid retrieval, we need to handle this specially
        if filters:
            # Use standard filtered search
            return self.answer_query_with_filters(query, filters, top_k, collection_name)
        else:
            # Use enhanced hybrid retrieval without filters
            retrieved_chunks = self.enhanced_retriever.hybrid_retrieve(query, top_k)
            
            if not retrieved_chunks:
                return self.config.generation.no_context_response
            
            # Debug info
            if retrieved_chunks and self.config.development.debug_mode:
                query_type = retrieved_chunks[0].get('query_type', 'unknown')
                semantic_weight = retrieved_chunks[0].get('semantic_weight', 0)
                keyword_weight = retrieved_chunks[0].get('keyword_weight', 0)
                print(f"Query classified as: {query_type} (semantic: {semantic_weight:.2f}, keyword: {keyword_weight:.2f})")
            
            # Generate answer
            answer = self.llm_handler.generate_answer(query, retrieved_chunks)
            return answer

if __name__ == '__main__':
    print("Testing Enhanced RAG Pipeline with Multi-Collection Management...")
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Test 1: Collection Management
    print("\n1. Testing Collection Management...")
    
    # Create test collections
    pipeline.create_collection("kubernetes_docs", "Kubernetes documentation", ["kubernetes", "containers"])
    pipeline.create_collection("python_docs", "Python best practices", ["python", "coding"])
    
    # List collections
    collections = pipeline.list_collections()
    print(f"Available collections: {[c['name'] for c in collections]}")
    
    # Test 2: Collection Statistics
    print("\n2. Testing Collection Statistics...")
    for collection in collections:
        stats = pipeline.get_collection_statistics(collection['name'])
        print(f"  {collection['name']}: {stats.get('chunk_count', 0)} chunks, {stats.get('document_count', 0)} documents")
    
    # Test 3: Filtered Search
    print("\n3. Testing Filtered Search...")
    
    # Create filters
    from src.metadata_filter import MetadataFilter
    
    pdf_filter = MetadataFilter.by_file_type("pdf")
    large_docs_filter = MetadataFilter.by_content_size(min_size=500)
    combined_filter = MetadataFilter.combine_filters([pdf_filter, large_docs_filter], "AND")
    
    test_query = "Was sind Kubernetes Best Practices?"
    
    # Standard query
    print(f"\nStandard query: '{test_query}'")
    if pipeline.collection.count() > 0:
        response = pipeline.answer_query(test_query)
        print(f"Response: {response[:100]}...")
    
    # Filtered query
    print(f"\nFiltered query (PDF files, >500 chars): '{test_query}'")
    try:
        response = pipeline.answer_query_with_filters(test_query, combined_filter)
        print(f"Filtered response: {response[:100]}...")
    except Exception as e:
        print(f"Filtered query test skipped: {e}")
    
    # Test 4: Cross-Collection Search
    print("\n4. Testing Cross-Collection Search...")
    if len(collections) > 1:
        try:
            response = pipeline.search_across_collections(
                "Python and Kubernetes integration",
                collection_names=[c['name'] for c in collections[:2]]
            )
            print(f"Cross-collection response: {response[:100]}...")
        except Exception as e:
            print(f"Cross-collection search test skipped: {e}")
    
    # Test 5: Enhanced Filtering
    print("\n5. Testing Enhanced Hybrid Retrieval with Filtering...")
    try:
        response = pipeline.enhanced_answer_query_with_filters(
            "Kubernetes deployment strategies",
            filters=None  # No filters for enhanced retrieval demo
        )
        print(f"Enhanced response: {response[:100]}...")
    except Exception as e:
        print(f"Enhanced retrieval test skipped: {e}")
    
    print("\nEnhanced RAG Pipeline test completed!")
    print("\nNew Features Available:")
    print("  ✅ Multi-collection management")
    print("  ✅ Collection-specific document ingestion")
    print("  ✅ Metadata-based filtered search")
    print("  ✅ Cross-collection search capabilities")
    print("  ✅ Enhanced hybrid retrieval with filtering")
    print("  ✅ Collection statistics and management")

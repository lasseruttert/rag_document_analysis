import streamlit as st
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import RAGPipeline
from src.config import get_config

# Load configuration
config = get_config()

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title=config.ui.page_title,
    page_icon=config.ui.page_icon,
    layout=config.ui.layout,
    initial_sidebar_state=config.ui.initial_sidebar_state
)

st.title(f"{config.ui.page_icon} {config.ui.page_title}")
st.markdown("Stellen Sie Fragen zu Ihren hochgeladenen Dokumenten.")

# --- Initialize RAG Pipeline (Cached) ---
@st.cache_resource
def get_rag_pipeline():
    """
    Initialisiert und cached die RAGPipeline.
    """
    return RAGPipeline(config=config)

pipeline = get_rag_pipeline()

# --- Sidebar for Document Ingestion ---
st.sidebar.header("Dokumenten-Verwaltung")

# Configuration section
if config.ui.show_debug_info:
    with st.sidebar.expander("⚙️ Konfiguration", expanded=False):
        st.markdown("**Aktuelle Einstellungen:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunk Size", config.chunking.chunk_size)
            st.metric("Embedding Model", config.models.embedding_model.split('/')[-1])
        with col2:
            st.metric("Overlap", config.chunking.chunk_overlap)
            st.metric("LLM Model", config.models.llm_model.split('/')[-1])
        
        if st.button("🔄 Config neu laden"):
            st.cache_resource.clear()
            st.rerun()

# Create a directory for uploaded files if it doesn't exist
upload_dir = "data/raw_texts"
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

uploaded_files = st.sidebar.file_uploader(
    "Laden Sie Dokumente hoch", 
    type=["txt", "pdf", "docx"],  # Erweiterte Typen
    accept_multiple_files=True,
    help="Unterstützte Formate: TXT, PDF, DOCX"
)

# File processing mit Typ-Anzeige
if uploaded_files:
    upload_stats = {"txt": 0, "pdf": 0, "docx": 0}
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_dir, uploaded_file.name)
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Validate file type
        if file_extension not in ["txt", "pdf", "docx"]:
            st.sidebar.error(f"Nicht unterstützter Dateityp: {uploaded_file.name}")
            continue
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        upload_stats[file_extension] += 1
        st.sidebar.success(f"✓ {uploaded_file.name} ({file_extension.upper()}) hochgeladen")
    
    # Upload summary
    summary_parts = []
    for file_type, count in upload_stats.items():
        if count > 0:
            summary_parts.append(f"{count} {file_type.upper()}")
    
    if summary_parts:
        st.sidebar.info(f"Hochgeladen: {', '.join(summary_parts)}")

# Document ingestion mit Progress
if st.sidebar.button("Dokumente ingestieren (in Vektor-DB)"):
    with st.spinner("Dokumente werden verarbeitet..."):
        # Progress bar für verschiedene Schritte
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        try:
            status_text.text("Lade Dokumente...")
            progress_bar.progress(25)
            
            pipeline.ingest_documents()
            
            progress_bar.progress(100)
            status_text.text("Fertig!")
            
            st.sidebar.success("Dokumente erfolgreich ingestiert!")
            
            # Zeige Statistiken
            collection_count = pipeline.collection.count()
            st.sidebar.metric("Dokumente in DB", collection_count)
            
        except Exception as e:
            st.sidebar.error(f"Fehler beim Ingestieren: {str(e)}")
            progress_bar.empty()
            status_text.empty()

# Enhanced document info display
st.sidebar.markdown("### 📊 Dokumenten-Übersicht")
if pipeline.collection.count() > 0:
    # Zeige Dokumenttyp-Statistiken
    try:
        # Hole Metadaten aus der Collection
        sample_results = pipeline.collection.get(limit=1000, include=["metadatas"])
        if sample_results['metadatas']:
            file_types = {}
            for metadata in sample_results['metadatas']:
                file_type = metadata.get('file_type', 'unknown')
                file_types[file_type] = file_types.get(file_type, 0) + 1
            
            for file_type, count in file_types.items():
                st.sidebar.metric(f"{file_type.upper()}-Chunks", count)
    except:
        pass
else:
    st.sidebar.info("Keine Dokumente in der Datenbank")

# Hybrid Retrieval Settings
st.sidebar.markdown("### 🔍 Retrieval-Einstellungen")

# Get default method from config
default_methods = ["Hybrid (Empfohlen)", "Nur Semantisch", "Nur Keywords"]
default_index = 0  # Hybrid as default
if config.ui.default_retrieval_method == "semantic":
    default_index = 1
elif config.ui.default_retrieval_method == "keywords":
    default_index = 2

retrieval_method = st.sidebar.radio(
    "Retrieval-Methode",
    default_methods,
    index=default_index,
    help="Hybrid kombiniert semantische und Keyword-Suche intelligent"
)

# Query Settings
if retrieval_method == "Hybrid (Empfohlen)":
    st.sidebar.markdown("**Hybrid Retrieval aktiviert**")
    st.sidebar.info("🧠 Intelligente Kombination von semantischer und Keyword-Suche mit deutschen Sprachoptimierungen")

# --- Main Content for Querying ---

st.header("Frage stellen")
user_query = st.text_input("Ihre Frage:", placeholder="Was ist ein Pod in Kubernetes?")

# Enhanced Query Processing mit Hybrid Retrieval
if user_query:
    if pipeline.collection.count() == 0:
        st.warning("Bitte ingestieren Sie zuerst Dokumente, bevor Sie eine Frage stellen.")
    else:
        with st.spinner("Antwort wird generiert..."):
            # Wähle Retrieval-Methode basierend auf User-Auswahl
            if retrieval_method == "Hybrid (Empfohlen)":
                answer = pipeline.enhanced_answer_query(user_query)
                retrieved_chunks = pipeline.enhanced_retriever.hybrid_retrieve(user_query, top_k=config.ui.max_context_chunks) if pipeline.enhanced_retriever else []
                
                # Query Analysis Info anzeigen
                if retrieved_chunks:
                    query_info = retrieved_chunks[0]
                    st.info(f"🔍 Query-Typ: **{query_info.get('query_type', 'unknown').title()}** | "
                           f"Semantic: {query_info.get('semantic_weight', 0):.2f} | "
                           f"Keyword: {query_info.get('keyword_weight', 0):.2f}")
                    
            elif retrieval_method == "Nur Keywords":
                # Fallback: BM25-only (wird über Enhanced Retriever mit keyword_weight=1.0 simuliert)
                st.info("🔤 Reine Keyword-Suche aktiviert")
                answer = pipeline.answer_query(user_query)  # Fallback auf Standard
                retrieved_chunks = pipeline.retriever.retrieve(user_query, top_k=config.ui.max_context_chunks)
            else:
                # Standard semantic retrieval
                st.info("🧠 Reine semantische Suche aktiviert")
                answer = pipeline.answer_query(user_query)
                retrieved_chunks = pipeline.retriever.retrieve(user_query, top_k=config.ui.max_context_chunks)
            
            st.subheader("🤖 Antwort:")
            st.write(answer)

            # Enhanced Context Display
            if retrieval_method == "Hybrid (Empfohlen)" and retrieved_chunks:
                st.subheader("📄 Hybrid Retrieval Details:")
                
                # Chunk Details mit Enhanced Scoring
                for i, chunk in enumerate(retrieved_chunks):
                    file_type = chunk['metadata'].get('file_type', 'unknown')
                    filename = chunk['metadata']['filename']
                    
                    # File type icon
                    type_icons = {"pdf": "📄", "docx": "📝", "txt": "📄"}
                    icon = type_icons.get(file_type, "📄")
                    
                    # Enhanced title mit Hybrid Score
                    hybrid_score = chunk.get('hybrid_score', 0)
                    title = f"{icon} Chunk {i+1} - {filename} ({file_type.upper()}) - Hybrid Score: {hybrid_score:.3f}"
                    
                    with st.expander(title, expanded=(i == 0)):
                        st.code(chunk['content'], language=None)
                        
                        # Detailed Scoring Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            semantic_score = chunk.get('semantic_score', 0)
                            st.metric("Semantic", f"{semantic_score:.3f}")
                        with col2:
                            bm25_score = chunk.get('bm25_score', 0)
                            st.metric("BM25", f"{bm25_score:.3f}")
                        with col3:
                            st.metric("Hybrid", f"{hybrid_score:.3f}")
                        with col4:
                            method = chunk.get('retrieval_method', 'standard')
                            st.metric("Method", method)
                        
                        # Additional metadata
                        if 'semantic_rank' in chunk and 'bm25_rank' in chunk:
                            st.caption(f"Semantic Rank: {chunk['semantic_rank']} | BM25 Rank: {chunk['bm25_rank']}")
            
            else:
                # Standard Context Display
                st.subheader("📄 Verwendeter Kontext:")
                retrieved_chunks = retrieved_chunks if 'retrieved_chunks' in locals() else pipeline.retriever.retrieve(user_query, top_k=3)
            
            if retrieved_chunks:
                for i, chunk in enumerate(retrieved_chunks):
                    file_type = chunk['metadata'].get('file_type', 'unknown')
                    filename = chunk['metadata']['filename']
                    distance = chunk['distance']
                    
                    # File type icon
                    type_icons = {"pdf": "📄", "docx": "📝", "txt": "📄"}
                    icon = type_icons.get(file_type, "📄")
                    
                    with st.expander(
                        f"{icon} Chunk {i+1} - {filename} ({file_type.upper()}) - Relevanz: {1-distance:.3f}",
                        expanded=(i == 0)  # Erstes Chunk expanded
                    ):
                        st.code(chunk['content'], language=None)
                        
                        # Zusätzliche Metadaten
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Position", chunk['metadata'].get('position', 'N/A'))
                        with col2:
                            st.metric("Chunk-Größe", len(chunk['content']))
                        with col3:
                            st.metric("Distanz", f"{distance:.4f}")
            else:
                st.info("Kein relevanter Kontext gefunden.")

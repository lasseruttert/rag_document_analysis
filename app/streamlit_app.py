import streamlit as st
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import RAGPipeline
from src.config import get_config
from src.metadata_filter import MetadataFilter, QueryFilter, CombinedFilter
from src.collection_manager import CollectionInfo, CollectionStats

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
st.sidebar.header("📂 Dokumenten-Verwaltung")

# === COLLECTION MANAGEMENT ===
with st.sidebar.expander("🗂️ Collection Management", expanded=False):
    st.markdown("**Collections verwalten:**")
    
    # List existing collections
    collections = pipeline.list_collections()
    if collections:
        current_collection_names = [c['name'] for c in collections]
        
        # Active collection selector
        active_collection = st.selectbox(
            "Aktive Collection:",
            current_collection_names,
            index=current_collection_names.index(pipeline.active_collection_name) if pipeline.active_collection_name in current_collection_names else 0,
            help="Wählen Sie die Collection für Queries aus"
        )
        
        # Set active collection if different
        if active_collection != pipeline.active_collection_name:
            if pipeline.set_active_collection(active_collection):
                st.success(f"✅ Aktive Collection: {active_collection}")
                st.rerun()
    
    # Create new collection
    st.markdown("**Neue Collection erstellen:**")
    col1, col2 = st.columns([3, 1])
    with col1:
        new_collection_name = st.text_input("Name:", key="new_collection")
    with col2:
        create_collection = st.button("➕", help="Collection erstellen")
    
    new_collection_desc = st.text_input("Beschreibung:", key="new_collection_desc")
    new_collection_tags = st.text_input("Tags (kommagetrennt):", key="new_collection_tags", help="z.B. kubernetes,docs")
    
    if create_collection and new_collection_name:
        tags = [tag.strip() for tag in new_collection_tags.split(",") if tag.strip()] if new_collection_tags else []
        success = pipeline.create_collection(new_collection_name, new_collection_desc, tags)
        if success:
            st.success(f"✅ Collection '{new_collection_name}' erstellt!")
            st.rerun()
        else:
            st.error(f"❌ Collection '{new_collection_name}' existiert bereits!")
    
    # Delete collection
    if collections:
        st.markdown("**Collection löschen:**")
        collection_to_delete = st.selectbox("Collection:", current_collection_names, key="delete_collection")
        delete_collection = st.button("🗑️ Löschen", help="Collection dauerhaft löschen")
        
        if delete_collection:
            if st.session_state.get('confirm_delete') != collection_to_delete:
                st.session_state.confirm_delete = collection_to_delete
                st.warning(f"⚠️ Klicken Sie erneut, um '{collection_to_delete}' zu löschen!")
            else:
                success = pipeline.delete_collection(collection_to_delete)
                if success:
                    st.success(f"✅ Collection '{collection_to_delete}' gelöscht!")
                    if 'confirm_delete' in st.session_state:
                        del st.session_state.confirm_delete
                    st.rerun()
                else:
                    st.error(f"❌ Fehler beim Löschen von '{collection_to_delete}'!")

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

# Collection-specific document ingestion
ingestion_col1, ingestion_col2 = st.sidebar.columns([3, 1])
with ingestion_col1:
    target_collection = st.selectbox(
        "Ziel-Collection:",
        [c['name'] for c in collections] if collections else ["Keine Collections verfügbar"],
        key="target_collection",
        help="Collection für neue Dokumente"
    )
with ingestion_col2:
    ingest_button = st.button("📥", help="Dokumente ingestieren")

# Document ingestion mit Progress
if ingest_button and target_collection != "Keine Collections verfügbar":
    with st.spinner("Dokumente werden verarbeitet..."):
        # Progress bar für verschiedene Schritte
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        try:
            status_text.text("Lade Dokumente...")
            progress_bar.progress(25)
            
            # Use collection-specific ingestion
            success = pipeline.ingest_documents_to_collection(target_collection)
            
            progress_bar.progress(100)
            status_text.text("Fertig!")
            
            if success:
                st.sidebar.success(f"Dokumente erfolgreich in '{target_collection}' ingestiert!")
                
                # Zeige Statistiken für die Collection
                stats = pipeline.get_collection_statistics(target_collection)
                if stats:
                    st.sidebar.metric(f"Chunks in {target_collection}", stats.get('chunk_count', 0))
            else:
                st.sidebar.warning("Keine neuen Dokumente gefunden oder Fehler beim Ingestieren.")
            
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

# === STATISTICS DASHBOARD ===
if collections:
    st.header("📊 Collection Dashboard")
    
    # Overview metrics
    total_collections = len(collections)
    total_chunks = sum(c.get('chunk_count', 0) for c in collections)
    total_docs = sum(c.get('document_count', 0) for c in collections)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Collections", total_collections)
    with col2:
        st.metric("Gesamt Dokumente", total_docs)
    with col3:
        st.metric("Gesamt Chunks", total_chunks)
    with col4:
        st.metric("Aktive Collection", pipeline.active_collection_name)
    
    # Detailed collection stats
    with st.expander("📈 Detailierte Collection-Statistiken", expanded=False):
        for collection in collections:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{collection['name']}**")
                    if collection['description']:
                        st.caption(collection['description'])
                    if collection['tags']:
                        st.caption(f"Tags: {', '.join(collection['tags'])}")
                with col2:
                    stats = pipeline.get_collection_statistics(collection['name'])
                    if stats:
                        st.metric("Chunks", stats.get('chunk_count', 0))
                        
                        # File type distribution
                        file_types = stats.get('file_types', {})
                        if file_types:
                            type_summary = ", ".join([f"{count} {ftype.upper()}" for ftype, count in file_types.items()])
                            st.caption(f"Dateitypen: {type_summary}")
                
                st.divider()

# === ADVANCED FILTERING INTERFACE ===
st.header("🔍 Erweiterte Suche")

# Enhanced Query Input with Preprocessing
st.markdown("### Intelligente Query-Eingabe")

# Query input with suggestions
query_col1, query_col2, query_col3 = st.columns([4, 1, 1])
with query_col1:
    user_query = st.text_input("Ihre Frage:", placeholder="Was ist ein Pod in Kubernetes?", key="main_query")
with query_col2:
    use_preprocessing = st.checkbox("Smart Query", value=True, help="Intelligente Query-Verarbeitung aktivieren")
with query_col3:
    use_filters = st.checkbox("Filter", help="Erweiterte Filterung aktivieren")

# Query Analysis and Suggestions
if user_query and use_preprocessing:
    try:
        # Analyze query intent and provide real-time feedback
        query_analysis = pipeline.analyze_query_intent(user_query)
        
        # Display query analysis in a compact info box
        analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
        with analysis_col1:
            intent_color = {
                'question': 'blue',
                'command': 'green', 
                'search': 'orange',
                'factual': 'blue',
                'procedural': 'violet',
                'comparison': 'red',
                'unknown': 'gray'
            }.get(query_analysis['intent'], 'gray')
            st.markdown(f"**Intent:** :{intent_color}[{query_analysis['intent'].upper()}]")
        
        with analysis_col2:
            confidence = query_analysis['confidence']
            confidence_text = "Hoch" if confidence > 0.8 else "Mittel" if confidence > 0.6 else "Niedrig"
            st.markdown(f"**Konfidenz:** {confidence_text} ({confidence:.1f})")
        
        with analysis_col3:
            keywords = query_analysis['extracted_info']['keywords']
            if keywords:
                st.markdown(f"**Keywords:** {', '.join(keywords[:3])}")
        
        # Show spelling suggestions if available
        spell_suggestions = query_analysis['extracted_info']['spell_suggestions']
        if spell_suggestions:
            st.info(f"Rechtschreibvorschläge: {', '.join(spell_suggestions)}")
        
    except Exception as e:
        st.warning(f"Query-Analyse nicht verfügbar: {str(e)}")

# Query Suggestions
if len(user_query) >= 3 and use_preprocessing:
    try:
        suggestions = pipeline.suggest_query_completions(user_query, limit=3)
        if suggestions and user_query.lower() not in [s.lower() for s in suggestions]:
            st.markdown("**Vorschläge:**")
            suggestion_cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                with suggestion_cols[i]:
                    if st.button(suggestion, key=f"suggestion_{i}", help="Klicken zum Übernehmen"):
                        st.session_state.main_query = suggestion
                        st.rerun()
    except Exception as e:
        pass  # Silently ignore suggestion errors

# Advanced Query Options
with st.expander("Erweiterte Query-Optionen", expanded=False):
    query_opt_col1, query_opt_col2 = st.columns(2)
    
    with query_opt_col1:
        enable_spell_check = st.checkbox("Rechtschreibkorrektur", value=True, help="Automatische Fehlerkorrektur")
        enable_expansion = st.checkbox("Query-Erweiterung", value=True, help="Synonyme und verwandte Begriffe hinzufügen")
    
    with query_opt_col2:
        intent_override = st.selectbox(
            "Intent überschreiben:", 
            ["Auto", "Question", "Command", "Search", "Factual"],
            help="Manuelle Intent-Festlegung"
        )
        processing_mode = st.radio(
            "Verarbeitungsmodus:",
            ["Intelligent", "Standard", "Exact Match"],
            help="Wähle die Query-Verarbeitungsstrategie"
        )

# Filtering interface
filters = None
if use_filters:
    with st.expander("🔧 Filter-Einstellungen", expanded=True):
        filter_tabs = st.tabs(["📁 Dateityp", "📏 Größe", "🏷️ Custom"])
        
        with filter_tabs[0]:  # File type filter
            st.markdown("**Nach Dateityp filtern:**")
            file_type_options = ["pdf", "docx", "txt"]
            selected_types = st.multiselect("Dateitypen:", file_type_options, help="Wähle einen oder mehrere Dateitypen")
            exclude_types = st.checkbox("Ausschließen statt einschließen", key="exclude_types")
            
            if selected_types:
                file_type_filter = MetadataFilter.by_file_type(selected_types, exclude=exclude_types)
            else:
                file_type_filter = None
        
        with filter_tabs[1]:  # Size filter
            st.markdown("**Nach Chunk-Größe filtern:**")
            col1, col2 = st.columns(2)
            with col1:
                min_size = st.number_input("Min. Zeichen:", min_value=0, value=0, step=100)
            with col2:
                max_size = st.number_input("Max. Zeichen:", min_value=0, value=5000, step=100)
            
            if min_size > 0 or max_size < 5000:
                size_filter = MetadataFilter.by_content_size(
                    min_size=min_size if min_size > 0 else None,
                    max_size=max_size if max_size < 5000 else None
                )
            else:
                size_filter = None
        
        with filter_tabs[2]:  # Custom filter
            st.markdown("**Custom Filter:**")
            custom_field = st.text_input("Feld:", placeholder="z.B. semantic_density")
            custom_value = st.text_input("Wert:", placeholder="z.B. 0.5")
            custom_operator = st.selectbox("Operator:", ["equals", "greater_than", "less_than", "contains"])
            
            if custom_field and custom_value:
                try:
                    # Try to convert to number if possible
                    try:
                        custom_value = float(custom_value)
                    except ValueError:
                        pass  # Keep as string
                    
                    from src.metadata_filter import FilterOperator
                    operator_map = {
                        "equals": FilterOperator.EQUALS,
                        "greater_than": FilterOperator.GREATER_THAN,
                        "less_than": FilterOperator.LESS_THAN,
                        "contains": FilterOperator.CONTAINS
                    }
                    
                    custom_filter = MetadataFilter.by_custom_field(
                        custom_field, custom_value, operator_map[custom_operator]
                    )
                except Exception as e:
                    st.error(f"Fehler beim Custom Filter: {e}")
                    custom_filter = None
            else:
                custom_filter = None
        
        # Combine filters
        active_filters = [f for f in [file_type_filter, size_filter, custom_filter] if f is not None]
        if active_filters:
            if len(active_filters) == 1:
                filters = active_filters[0]
            else:
                combine_operator = st.selectbox("Filter-Kombination:", ["AND", "OR"], key="filter_combine")
                filters = MetadataFilter.combine_filters(active_filters, combine_operator)
            
            st.success(f"✅ {len(active_filters)} Filter aktiv")
        else:
            st.info("ℹ️ Keine Filter ausgewählt")

# === QUERY OPTIONS ===
query_options_col1, query_options_col2, query_options_col3 = st.columns(3)

with query_options_col1:
    search_scope = st.radio(
        "Suchbereich:",
        ["Aktive Collection", "Alle Collections", "Custom Collections"],
        help="Wo soll gesucht werden?"
    )

with query_options_col2:
    query_method = st.radio(
        "Suchmethode:",
        ["Hybrid (Empfohlen)", "Semantisch", "Mit Filtern"],
        help="Wähle die Suchmethode"
    )

with query_options_col3:
    top_k = st.slider("Max. Ergebnisse:", 1, 20, config.retrieval.default_top_k)

# Custom collection selection for cross-collection search
selected_collections = None
if search_scope == "Custom Collections" and collections:
    selected_collections = st.multiselect(
        "Collections auswählen:",
        [c['name'] for c in collections],
        default=[pipeline.active_collection_name] if pipeline.active_collection_name else []
    )

# Enhanced Query Processing with Multi-Collection and Filtering Support
if user_query:
    # Check if any collections have data
    has_data = any(c.get('chunk_count', 0) > 0 for c in collections) if collections else False
    
    if not has_data:
        st.warning("Bitte ingestieren Sie zuerst Dokumente in eine Collection, bevor Sie eine Frage stellen.")
    else:
        with st.spinner("Antwort wird generiert..."):
            retrieved_chunks = []
            
            # Apply intelligent preprocessing if enabled
            final_query = user_query
            query_info = None
            
            if use_preprocessing and processing_mode == "Intelligent":
                try:
                    # Use enhanced preprocessing
                    result = pipeline.enhanced_answer_query_with_preprocessing(
                        user_query,
                        top_k=top_k,
                        collection_name=selected_collections[0] if search_scope == "Custom Collections" and selected_collections else None,
                        use_spell_check=enable_spell_check,
                        use_expansion=enable_expansion
                    )
                    
                    answer = result['answer']
                    query_info = result['query_analysis']
                    final_query = query_info['processed_query']
                    
                    # Show preprocessing info
                    if query_info['spell_corrections']:
                        st.success(f"Query verbessert: '{user_query}' → '{final_query}'")
                    
                    st.info(f"Intelligente Verarbeitung: {query_info['intent']} (Konfidenz: {query_info['confidence']:.2f})")
                    
                except Exception as e:
                    st.warning(f"Intelligente Verarbeitung fehlgeschlagen, verwende Standard: {str(e)}")
                    final_query = user_query
                    query_info = None
            
            # If not using intelligent processing or it failed, use standard processing
            if not query_info:
                # Determine query processing based on options
                try:
                    if search_scope == "Alle Collections":
                        # Cross-collection search
                        if query_method == "Mit Filtern" and filters:
                            answer = pipeline.search_across_collections(
                                final_query, 
                                filters=filters,
                                total_results=top_k
                            )
                            st.info("Cross-Collection Suche mit Filtern")
                        else:
                            answer = pipeline.search_across_collections(final_query, total_results=top_k)
                            st.info("Cross-Collection Suche")
                    
                    elif search_scope == "Custom Collections" and selected_collections:
                        # Custom collection search
                        if query_method == "Mit Filtern" and filters:
                            answer = pipeline.search_across_collections(
                                final_query,
                                collection_names=selected_collections,
                                filters=filters,
                                total_results=top_k
                            )
                            st.info(f"Custom Collection Suche mit Filtern: {', '.join(selected_collections)}")
                        else:
                            answer = pipeline.search_across_collections(
                                final_query,
                                collection_names=selected_collections,
                                total_results=top_k
                            )
                            st.info(f"Custom Collection Suche: {', '.join(selected_collections)}")
                    
                    else:
                        # Single collection search (active collection)
                        if query_method == "Mit Filtern" and filters:
                            answer = pipeline.answer_query_with_filters(final_query, filters=filters, top_k=top_k)
                            st.info(f"Filtered Suche in '{pipeline.active_collection_name}'")
                    
                        elif query_method == "Hybrid (Empfohlen)":
                            if filters:
                                answer = pipeline.enhanced_answer_query_with_filters(final_query, filters=filters, top_k=top_k)
                            else:
                                answer = pipeline.enhanced_answer_query(final_query, top_k=top_k)
                                # Get enhanced retrieval info
                                if pipeline.enhanced_retriever:
                                    retrieved_chunks = pipeline.enhanced_retriever.hybrid_retrieve(final_query, top_k)
                                    if retrieved_chunks:
                                        query_info_retrieval = retrieved_chunks[0]
                                        st.info(f"Hybrid Query-Typ: **{query_info_retrieval.get('query_type', 'unknown').title()}** | "
                                               f"Semantic: {query_info_retrieval.get('semantic_weight', 0):.2f} | "
                                               f"Keyword: {query_info_retrieval.get('keyword_weight', 0):.2f}")
                            
                            st.info(f"Hybrid Retrieval in '{pipeline.active_collection_name}'")
                        
                        else:
                            # Standard semantic search
                            if filters:
                                answer = pipeline.answer_query_with_filters(final_query, filters=filters, top_k=top_k)
                                st.info(f"Semantische Suche mit Filtern in '{pipeline.active_collection_name}'")
                            else:
                                answer = pipeline.answer_query(final_query, top_k=top_k)
                                retrieved_chunks = pipeline.retriever.retrieve(final_query, top_k=top_k)
                                st.info(f"Semantische Suche in '{pipeline.active_collection_name}'")
                
                except Exception as e:
                    st.error(f"Fehler bei der Abfrage: {str(e)}")
                    answer = "Entschuldigung, es gab einen Fehler bei der Verarbeitung Ihrer Anfrage."
            
            st.subheader("🤖 Antwort:")
            st.write(answer)

            # Enhanced Context Display with Collection Info
            if config.ui.show_debug_info or True:  # Always show for enhanced experience
                
                # Show context details based on search method
                if query_method == "Hybrid (Empfohlen)" and retrieved_chunks:
                    st.subheader("📄 Hybrid Retrieval Context:")
                    
                    # Chunk Details mit Enhanced Scoring
                    for i, chunk in enumerate(retrieved_chunks):
                        file_type = chunk['metadata'].get('file_type', 'unknown')
                        filename = chunk['metadata']['filename']
                        collection_name = chunk['metadata'].get('collection_name', pipeline.active_collection_name)
                        
                        # File type icon
                        type_icons = {"pdf": "📄", "docx": "📝", "txt": "📄"}
                        icon = type_icons.get(file_type, "📄")
                        
                        # Enhanced title mit Hybrid Score
                        hybrid_score = chunk.get('hybrid_score', 0)
                        title = f"{icon} [{collection_name}] {filename} ({file_type.upper()}) - Score: {hybrid_score:.3f}"
                        
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
                                method = chunk.get('retrieval_method', 'hybrid')
                                st.metric("Method", method)
                            
                            # Additional metadata with collection info
                            metadata_cols = st.columns(4)
                            with metadata_cols[0]:
                                st.metric("Collection", collection_name)
                            with metadata_cols[1]:
                                st.metric("Position", chunk['metadata'].get('position', 'N/A'))
                            with metadata_cols[2]:
                                st.metric("Chunk Size", len(chunk['content']))
                            with metadata_cols[3]:
                                if 'semantic_rank' in chunk and 'bm25_rank' in chunk:
                                    st.caption(f"Ranks: S{chunk['semantic_rank']} | B{chunk['bm25_rank']}")
                
                else:
                    # Enhanced Standard Context Display with Collection Support
                    context_title = "📄 Kontext-Details:"
                    if search_scope != "Aktive Collection":
                        context_title = f"📄 {search_scope} Kontext:"
                    
                    st.subheader(context_title)
                    
                    # Try to get context chunks for display
                    if not retrieved_chunks and search_scope == "Aktive Collection":
                        try:
                            retrieved_chunks = pipeline.retriever.retrieve(user_query, top_k=min(top_k, 5))
                        except:
                            retrieved_chunks = []
                    
                    if retrieved_chunks:
                        for i, chunk in enumerate(retrieved_chunks):
                            file_type = chunk['metadata'].get('file_type', 'unknown')
                            filename = chunk['metadata']['filename']
                            collection_name = chunk['metadata'].get('collection_name', pipeline.active_collection_name)
                            
                            # File type icon
                            type_icons = {"pdf": "📄", "docx": "📝", "txt": "📄"}
                            icon = type_icons.get(file_type, "📄")
                            
                            # Handle different scoring types
                            score_info = ""
                            if 'distance' in chunk:
                                relevance = 1 - chunk['distance']
                                score_info = f"Relevanz: {relevance:.3f}"
                            elif 'hybrid_score' in chunk:
                                score_info = f"Score: {chunk['hybrid_score']:.3f}"
                            
                            title = f"{icon} [{collection_name}] {filename} ({file_type.upper()})"
                            if score_info:
                                title += f" - {score_info}"
                            
                            with st.expander(title, expanded=(i == 0)):
                                st.code(chunk['content'], language=None)
                                
                                # Enhanced metadata display
                                metadata_cols = st.columns(5)
                                with metadata_cols[0]:
                                    st.metric("Collection", collection_name)
                                with metadata_cols[1]:
                                    st.metric("File Type", file_type.upper())
                                with metadata_cols[2]:
                                    st.metric("Position", chunk['metadata'].get('position', 'N/A'))
                                with metadata_cols[3]:
                                    st.metric("Size", len(chunk['content']))
                                with metadata_cols[4]:
                                    if 'distance' in chunk:
                                        st.metric("Distance", f"{chunk['distance']:.4f}")
                                    elif 'hybrid_score' in chunk:
                                        st.metric("Score", f"{chunk['hybrid_score']:.3f}")
                                
                                # Show filter match info if filters were used
                                if filters:
                                    st.caption("✅ Entspricht den angewendeten Filtern")
                    
                    else:
                        if filters:
                            st.warning("🔍 Keine Ergebnisse entsprechen den angewendeten Filtern. Versuchen Sie weniger restriktive Filter.")
                        else:
                            st.info("ℹ️ Kein relevanter Kontext gefunden.")

import streamlit as st
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import RAGPipeline

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="RAG Document Analysis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 RAG Document Analysis")
st.markdown("Stellen Sie Fragen zu Ihren hochgeladenen Dokumenten.")

# --- Initialize RAG Pipeline (Cached) ---
@st.cache_resource
def get_rag_pipeline():
    """
    Initialisiert und cached die RAGPipeline.
    """
    return RAGPipeline()

pipeline = get_rag_pipeline()

# --- Sidebar for Document Ingestion ---
st.sidebar.header("Dokumenten-Verwaltung")

# Create a directory for uploaded files if it doesn't exist
upload_dir = "data/raw_texts"
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

uploaded_files = st.sidebar.file_uploader(
    "Laden Sie hier .txt-Dateien hoch", 
    type="txt", 
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Datei '{uploaded_file.name}' erfolgreich hochgeladen.")

if st.sidebar.button("Dokumente ingestieren (in Vektor-DB)"):
    with st.spinner("Dokumente werden verarbeitet und in die Vektor-Datenbank geladen..."):
        pipeline.ingest_documents()
    st.sidebar.success("Dokumente erfolgreich ingestiert!")

st.sidebar.markdown("--- ")
st.sidebar.info(f"Aktuell in der Vektor-DB: {pipeline.collection.count()} Dokumente")

# --- Main Content for Querying ---

st.header("Frage stellen")
user_query = st.text_input("Ihre Frage:", placeholder="Was ist ein Pod in Kubernetes?")

if user_query:
    if pipeline.collection.count() == 0:
        st.warning("Bitte ingestieren Sie zuerst Dokumente, bevor Sie eine Frage stellen.")
    else:
        with st.spinner("Antwort wird generiert..."):
            answer = pipeline.answer_query(user_query)
            st.subheader("Antwort:")
            st.write(answer)

            # Display retrieved context (optional, for debugging/transparency)
            st.subheader("Verwendeter Kontext (Top 3):")
            # Retrieve context again to display, as answer_query doesn't return it
            retrieved_chunks = pipeline.retriever.retrieve(user_query, top_k=3)
            if retrieved_chunks:
                for i, chunk in enumerate(retrieved_chunks):
                    st.markdown(f"**Chunk {i+1} (Distanz: {chunk['distance']:.4f}) aus {chunk['metadata']['filename']}:**")
                    st.code(chunk['content'])
            else:
                st.info("Kein relevanter Kontext gefunden.")

from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict

class LLMHandler:
    def __init__(self, model_name: str = 'google/flan-t5-base'):
        """
        Initialisiert den LLMHandler.

        Args:
            model_name: Name des T5-Modells von Hugging Face.
        """
        print(f"Loading model: {model_name}...")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        print("Model loaded successfully.")

    def generate_answer(self, query: str, context_chunks: List[Dict[str, any]]) -> str:
        """
        Generiert eine Antwort basierend auf der Anfrage und dem Kontext.

        Args:
            query: Die ursprüngliche Anfrage des Benutzers.
            context_chunks: Eine Liste von Kontext-Chunks vom Retriever.

        Returns:
            Die vom LLM generierte Antwort.
        """
        if not context_chunks:
            return "Ich konnte keine relevanten Informationen finden, um diese Frage zu beantworten."

        # Kontext aus den Chunks zusammenstellen
        context = "\n\n".join([chunk['content'] for chunk in context_chunks])

        # Prompt-Template erstellen
        prompt = f"""Beantworte die Frage basierend ausschließlich auf dem gegebenen Kontext. Wenn die Antwort nicht im Kontext enthalten ist, sage das deutlich.

Kontext: {context}

Frage: {query}

Antwort:"""

        print("\nGenerating answer...")
        print(f"Prompt length: {len(prompt)} characters")

        # Text generieren
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_beams=5, # Verbessert die Qualität der Ausgabe
            early_stopping=True
        )
        
        # Antwort dekodieren
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

if __name__ == '__main__':
    # Setup für den Test
    from src.embeddings import EmbeddingManager
    from src.vectorstore import VectorStoreManager
    from src.retriever import Retriever
    from src.text_processor import process_documents

    # 1. Komponenten initialisieren
    print("Initializing components...")
    embed_manager = EmbeddingManager()
    vector_store = VectorStoreManager()
    collection = vector_store.create_or_get_collection("llm_test")

    # 2. Daten laden und in Vektor-DB speichern (falls nötig)
    if collection.count() == 0:
        print("Populating vector store...")
        document_chunks = process_documents('data/raw_texts')
        chunks_with_embeddings = embed_manager.generate_embeddings(document_chunks)
        vector_store.add_documents(chunks_with_embeddings)

    # 3. Retriever und LLMHandler initialisieren
    retriever = Retriever(embed_manager, vector_store)
    llm_handler = LLMHandler()

    # 4. Testanfrage
    test_query = "Was ist ein Pod in Kubernetes und wofür wird er verwendet?"
    print(f"\n--- Testing Query: '{test_query}' ---")

    # 5. Kontext abrufen
    retrieved_context = retriever.retrieve(test_query, top_k=3)
    print("\nRetrieved context:")
    for i, chunk in enumerate(retrieved_context):
        print(f"  {i+1}. {chunk['content'][:100]}...")

    # 6. Antwort generieren
    final_answer = llm_handler.generate_answer(test_query, retrieved_context)

    print("\n--- Final Answer ---")
    print(final_answer)
    print("-" * 20)

from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict, Optional
from src.config import get_config, GenerationConfig

class LLMHandler:
    def __init__(self, 
                 model_name: Optional[str] = None,
                 config: Optional[GenerationConfig] = None):
        """
        Initialisiert den LLMHandler.

        Args:
            model_name: Name des T5-Modells von Hugging Face (uses config if None).
            config: GenerationConfig instance (uses global config if None).
        """
        # Load configuration
        if config is None:
            full_config = get_config()
            config = full_config.generation
            model_name = model_name or full_config.models.llm_model
        else:
            model_name = model_name or get_config().models.llm_model
        
        self.config = config
        
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
            return self.config.no_context_response

        # Kontext aus den Chunks zusammenstellen
        context = "\n\n".join([chunk['content'] for chunk in context_chunks])

        # Prompt-Template erstellen (using configured system prompt)
        if self.config.answer_language.lower() == "german":
            prompt = f"""{self.config.system_prompt}

Kontext: {context}

Frage: {query}

Antwort:"""
        else:
            prompt = f"""Answer the question based exclusively on the given context. If the answer is not contained in the context, say so clearly.

Context: {context}

Question: {query}

Answer:"""

        print("\nGenerating answer...")
        print(f"Prompt length: {len(prompt)} characters")

        # Text generieren mit konfigurierten Parametern
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=self.config.max_input_tokens, 
            truncation=True
        )
        
        outputs = self.model.generate(
            **inputs,
            max_length=self.config.max_output_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
            num_beams=self.config.num_beams,
            early_stopping=self.config.early_stopping
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

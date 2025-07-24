import os
from typing import List, Dict, Tuple

def load_text_files(directory_path: str) -> List[Tuple[str, str]]:
    """
    Lädt alle .txt-Dateien aus einem Verzeichnis.

    Args:
        directory_path: Pfad zum Verzeichnis mit den Textdateien.

    Returns:
        Eine Liste von Tupeln, wobei jedes Tupel den Dateinamen und den Inhalt enthält.
    """
    texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                texts.append((filename, f.read()))
    return texts

def chunk_text(
    text: str,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict[str, any]]:
    """
    Teilt einen Text in überlappende Chunks auf.

    Args:
        text: Der Eingabetext.
        filename: Der Dateiname des ursprünglichen Textes.
        chunk_size: Die maximale Zeichenlänge eines Chunks.
        chunk_overlap: Die Anzahl der überlappenden Zeichen zwischen Chunks.

    Returns:
        Eine Liste von Dictionaries, wobei jedes Dictionary einen Chunk und seine Metadaten enthält.
    """
    if not text:
        return []

    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = start + chunk_size
        chunk_content = text[start:end]
        
        chunks.append({
            "content": chunk_content,
            "metadata": {
                "filename": filename,
                "position": start,
                "chunk_id": f"{filename}-{chunk_id}"
            }
        })
        
        start += chunk_size - chunk_overlap
        chunk_id += 1
        
    return chunks

def process_documents(
    directory_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict[str, any]]:
    """
    Verarbeitet alle Textdokumente aus einem Verzeichnis zu Chunks.

    Args:
        directory_path: Pfad zum Verzeichnis mit den Textdateien.
        chunk_size: Die maximale Zeichenlänge eines Chunks.
        chunk_overlap: Die Anzahl der überlappenden Zeichen zwischen Chunks.

    Returns:
        Eine Liste aller Chunks aus allen Dokumenten.
    """
    all_chunks = []
    loaded_files = load_text_files(directory_path)
    
    for filename, content in loaded_files:
        chunks = chunk_text(
            text=content,
            filename=filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        all_chunks.extend(chunks)
        
    return all_chunks

if __name__ == '__main__':
    # Beispiel für die Verwendung
    data_path = 'data/raw_texts'
    
    # Sicherstellen, dass das Verzeichnis existiert
    if not os.path.exists(data_path):
        print(f"Error: Directory not found at '{data_path}'")
    else:
        # Laden der Texte zum Testen
        raw_texts = load_text_files(data_path)
        if not raw_texts:
            print(f"No .txt files found in '{data_path}'.")
        else:
            print(f"Found {len(raw_texts)} text file(s).")
            for name, text_content in raw_texts:
                print(f"--- Processing {name} ---")
                # Chunking des ersten gefundenen Textes
                document_chunks = chunk_text(text_content, name)
                
                if not document_chunks:
                    print("No chunks were created.")
                else:
                    print(f"Created {len(document_chunks)} chunks.")
                    # Ausgabe des ersten und letzten Chunks zur Überprüfung
                    print("First chunk:")
                    print(document_chunks[0])
                    print("\nLast chunk:")
                    print(document_chunks[-1])
                    print("-" * 20)

            # Gesamte Verarbeitung testen
            all_processed_chunks = process_documents(data_path)
            print(f"\nTotal processed chunks from all files: {len(all_processed_chunks)}")
            if all_processed_chunks:
                print("Sample chunk from total processing:")
                print(all_processed_chunks[0])
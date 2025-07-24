import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# PDF parsing imports
try:
    import pdfplumber
    PDF_PLUMBER_AVAILABLE = True
except ImportError:
    PDF_PLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# DOCX parsing imports
try:
    from docx import Document
    from docx.table import Table
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File size limit (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

def load_pdf_files(directory_path: str) -> List[Tuple[str, str, str]]:
    """
    Lädt alle .pdf-Dateien aus einem Verzeichnis.
    
    Args:
        directory_path: Pfad zum Verzeichnis mit den PDF-Dateien.
    
    Returns:
        List[Tuple[filename, content, file_type]]
    """
    if not PDF_PLUMBER_AVAILABLE and not PYMUPDF_AVAILABLE:
        logger.warning("Weder pdfplumber noch PyMuPDF verfügbar. PDF-Support deaktiviert.")
        return []
    
    pdf_files = []
    directory = Path(directory_path)
    
    if not directory.exists():
        logger.error(f"Directory not found: {directory_path}")
        return []
    
    for pdf_path in directory.glob("*.pdf"):
        if not validate_file_size(pdf_path):
            continue
            
        try:
            content = ""
            
            # Primäre Methode: pdfplumber (bessere Textextraktion)
            if PDF_PLUMBER_AVAILABLE:
                content = extract_text_with_pdfplumber(pdf_path)
            
            # Fallback: PyMuPDF für komplexe PDFs
            if not content.strip() and PYMUPDF_AVAILABLE:
                logger.info(f"Using PyMuPDF fallback for {pdf_path.name}")
                content = extract_text_with_pymupdf(pdf_path)
            
            if content.strip():
                pdf_files.append((pdf_path.name, content, "pdf"))
                logger.info(f"Successfully processed PDF: {pdf_path.name} ({len(content)} chars)")
            else:
                logger.warning(f"No text extracted from {pdf_path.name}")
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path.name}: {e}")
            continue
    
    return pdf_files

def extract_text_with_pdfplumber(pdf_path: Path) -> str:
    """Extrahiert Text mit pdfplumber (behält Tabellen-Struktur)."""
    full_text = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # Text extrahieren
                    text = page.extract_text()
                    if text:
                        full_text.append(f"[Seite {page_num}]\n{text}")
                    
                    # Tabellen separat extrahieren
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table:
                            table_text = format_table_as_text(table)
                            if table_text:
                                full_text.append(f"[Seite {page_num}, Tabelle {table_idx + 1}]\n{table_text}")
                                
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num} from {pdf_path.name}: {e}")
                    continue
    except Exception as e:
        logger.error(f"pdfplumber failed for {pdf_path.name}: {e}")
        return ""
    
    return "\n\n".join(full_text)

def format_table_as_text(table: List[List[str]]) -> str:
    """Konvertiert Tabelle zu strukturiertem Text."""
    if not table or not table[0]:
        return ""
    
    try:
        # Header-Zeile
        headers = [str(cell) if cell is not None else "" for cell in table[0]]
        if not any(headers):  # Skip empty headers
            return ""
            
        formatted_rows = [" | ".join(headers)]
        formatted_rows.append("-" * len(" | ".join(headers)))
        
        # Daten-Zeilen
        for row in table[1:]:
            if row:
                formatted_row = [str(cell) if cell is not None else "" for cell in row]
                formatted_rows.append(" | ".join(formatted_row))
        
        return "\n".join(formatted_rows)
    except Exception as e:
        logger.warning(f"Error formatting table: {e}")
        return ""

def extract_text_with_pymupdf(pdf_path: Path) -> str:
    """Fallback-Extraktion mit PyMuPDF."""
    full_text = []
    
    try:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(pdf_document.page_count):
            try:
                page = pdf_document[page_num]
                text = page.get_text()
                if text.strip():
                    full_text.append(f"[Seite {page_num + 1}]\n{text}")
            except Exception as e:
                logger.warning(f"Error extracting page {page_num + 1} from {pdf_path.name}: {e}")
                continue
        pdf_document.close()
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed for {pdf_path.name}: {e}")
        return ""
    
    return "\n\n".join(full_text)

def load_docx_files(directory_path: str) -> List[Tuple[str, str, str]]:
    """
    Lädt alle .docx-Dateien aus einem Verzeichnis.
    
    Args:
        directory_path: Pfad zum Verzeichnis mit den DOCX-Dateien.
    
    Returns:
        List[Tuple[filename, content, file_type]]
    """
    if not DOCX_AVAILABLE:
        logger.warning("python-docx nicht verfügbar. DOCX-Support deaktiviert.")
        return []
    
    docx_files = []
    directory = Path(directory_path)
    
    if not directory.exists():
        logger.error(f"Directory not found: {directory_path}")
        return []
    
    for docx_path in directory.glob("*.docx"):
        # Skip temporary files (~$filename.docx)
        if docx_path.name.startswith("~$"):
            continue
            
        if not validate_file_size(docx_path):
            continue
            
        try:
            content = extract_text_from_docx(docx_path)
            if content.strip():
                docx_files.append((docx_path.name, content, "docx"))
                logger.info(f"Successfully processed DOCX: {docx_path.name} ({len(content)} chars)")
            else:
                logger.warning(f"No text extracted from {docx_path.name}")
                
        except Exception as e:
            logger.error(f"Error processing DOCX {docx_path.name}: {e}")
            continue
    
    return docx_files

def extract_text_from_docx(docx_path: Path) -> str:
    """Extrahiert Text und Tabellen aus DOCX-Datei."""
    try:
        document = Document(docx_path)
        full_text = []
        
        # Iteriere über alle Elemente (Paragraphs und Tabellen) in Reihenfolge
        for element in document.element.body:
            if isinstance(element, CT_P):
                # Paragraph
                paragraph = next((p for p in document.paragraphs if p._element == element), None)
                if paragraph:
                    text = paragraph.text.strip()
                    if text:
                        # Formatierung beibehalten
                        if paragraph.style.name.startswith('Heading'):
                            full_text.append(f"\n## {text}\n")
                        else:
                            full_text.append(text)
                            
            elif isinstance(element, CT_Tbl):
                # Tabelle
                table = next((t for t in document.tables if t._element == element), None)
                if table:
                    table_text = extract_table_from_docx(table)
                    if table_text:
                        full_text.append(f"\n[Tabelle]\n{table_text}\n")
        
        return "\n".join(full_text)
        
    except Exception as e:
        logger.error(f"Error extracting from DOCX {docx_path.name}: {e}")
        return ""

def extract_table_from_docx(table: Table) -> str:
    """Extrahiert Tabelle aus DOCX als strukturierten Text."""
    if not table.rows:
        return ""
    
    try:
        formatted_rows = []
        
        for row_idx, row in enumerate(table.rows):
            row_cells = []
            for cell in row.cells:
                cell_text = cell.text.strip().replace("\n", " ")
                row_cells.append(cell_text or "")
            
            formatted_rows.append(" | ".join(row_cells))
            
            # Header-Separator nach erster Zeile
            if row_idx == 0 and len(table.rows) > 1:
                separator = " | ".join(["-" * max(3, len(cell)) for cell in row_cells])
                formatted_rows.append(separator)
        
        return "\n".join(formatted_rows)
    except Exception as e:
        logger.warning(f"Error extracting table from DOCX: {e}")
        return ""

def validate_file_size(file_path: Path) -> bool:
    """Validiert Dateigröße."""
    try:
        size = file_path.stat().st_size
        if size > MAX_FILE_SIZE:
            logger.warning(f"File {file_path.name} too large: {size/1024/1024:.1f}MB (max: {MAX_FILE_SIZE/1024/1024}MB)")
            return False
        return True
    except Exception as e:
        logger.error(f"Error checking file size for {file_path}: {e}")
        return False

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
        file_type: Der Dateityp (txt, pdf, docx).

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
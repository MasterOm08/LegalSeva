from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

class LegalTextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        # Optimized separators for legal agreements and contracts
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\nSECTION ", 
                "\nARTICLE ", 
                "\n\n", 
                "\n", 
                ". ", 
                " "
            ]
        )

    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_chunks = self.text_splitter.split_text(text)
        
        chunks = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_meta = metadata.copy()
            chunk_meta["chunk_index"] = idx
            
            chunks.append({
                "chunk_id": f"{metadata['filename']}_chunk_{idx}",
                "text": chunk_text,
                "metadata": chunk_meta
            })
            
        return chunks

# --- Wrapper for FastAPI (src/main.py) compatibility ---
_default_chunker = LegalTextChunker()

def chunk_text(text: str, filename: str = "") -> List[str]:
    """
    Wrapper function expected by src/main.py.
    Chunks raw text using LegalTextChunker and returns a list of string chunks.
    """
    metadata = {"filename": filename} if filename else {}
    structured_chunks = _default_chunker.create_chunks(text, metadata=metadata)
    
    # Extract just the raw string text for ChromaDB ingestion
    return [chunk["text"] for chunk in structured_chunks]
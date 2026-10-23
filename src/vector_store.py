import os
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config import settings

class VectorStoreManager:
    def __init__(self):
        # Local HuggingFace embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        self.persist_dir = settings.CHROMA_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.vector_db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name="legal_contracts"
        )

    def upsert_chunks(self, chunks: List[Dict[str, Any]]):
        documents = [
            Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"],
                id=chunk["chunk_id"]
            )
            for chunk in chunks
        ]
        self.vector_db.add_documents(documents)

    def similarity_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        docs = self.vector_db.similarity_search(query, k=top_k)
        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]
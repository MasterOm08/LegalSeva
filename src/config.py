import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    APP_NAME: str = "Legal Contract AI Reviewer"
    
    # Free Groq LLM Settings
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = "openai/gpt-oss-120b"  # Active replacement model on Groq
    
    # Local HuggingFace Embedding Model
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    
    # Local Chroma DB Path
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
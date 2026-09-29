import http

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    OPENAI_API_KEY : str
    OPENAI_BASE_URL : str = "https://openai.vocareum.com/v1"
    QDRANT_URL : str = "http://localhost:6333"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

my_settings = Settings()

# ─── Configuration ──────────────────────────────────────────────────────

CORPUS_DIR=Path(__file__).parent / "data" / "corpus"
DATA_DIR=Path(__file__).parent / "data"
COLLECTION_NAME="mp2_sherlock"
EMBEDDING_MODEL="text-embedding-3-small"
EMBEDDING_DIM=1536
CHAT_MODEL="gpt-4o-mini"
TARGET_CHUNK_SIZE=500   # characters
CHUNK_OVERLAP=80    # characters
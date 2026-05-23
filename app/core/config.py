from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "Enterprise RAG Platform"
    APP_ENV: str = "development"
    DEBUG: bool = True

    GOOGLE_API_KEY: str | None = None
    LLM_PROVIDER: str = "gemini"
    LLM_MODEL: str = "gemini-2.5-flash"
    
    EMBEDDING_PROVIDER: str = "gemini"
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    EMBEDDING_DIMENSION: int = 3072

    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "rag_platform"
    POSTGRES_USER: str = "rag_user"
    POSTGRES_PASSWORD: str
    DATABASE_URL: str

    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_PREFIX: str = "rag_index"

    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 80

    DENSE_TOP_K: int = 20
    SPARSE_TOP_K: int = 20
    RERANK_TOP_K: int = 5

    MAX_UPLOAD_MB: int = 20
    UPLOAD_DIR: str = "uploads"

    EVAL_DOCUMENT_ID: str | None = None

    MIN_HYBRID_SCORE: float = 0.45

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
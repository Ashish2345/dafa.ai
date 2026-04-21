"""
Application configuration using Pydantic Settings.

Configuration is loaded from environment variables and .env files.
"""

from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models.enums import Environment, StorageBackend, RetrievalStrategyType


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Let the `mode="before"` validators parse comma-separated strings for
        # List fields (cors_origins, allowed_extensions, ocr_languages) instead
        # of pydantic-settings trying to JSON-decode them first.
        enable_decoding=False,
    )

    # Application
    app_name: str = Field(default="Document Parser Service", description="Application name")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Environment")
    debug: bool = Field(default=False, description="Debug mode")
    secret_key: str = Field(
        default="change-me-in-production-minimum-32-characters-long",
        description="Secret key for signing",
        min_length=32,
    )
    api_v1_prefix: str = Field(default="/api/v1", description="API v1 URL prefix")

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port", ge=1, le=65535)
    workers: int = Field(default=4, description="Number of workers", ge=1, le=32)

    # CORS — browsers send the page origin (e.g. https://*.ngrok-free.app); it must be listed or match cors_allow_origin_regex
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "https://app.merodafa.com",
            "https://merodafa.com"
        ],
        description="Allowed CORS origins (exact match). Add more via CORS_ORIGINS=comma,separated",
    )
    cors_allow_credentials: bool = Field(default=True, description="Allow credentials in CORS")
    cors_allow_origin_regex: Optional[str] = Field(
        default=(
            r"https://[a-zA-Z0-9][a-zA-Z0-9.-]*\.ngrok-free\.app$|"
            r"https://[a-zA-Z0-9][a-zA-Z0-9.-]*\.ngrok\.io$|"
            r"https://[a-zA-Z0-9][a-zA-Z0-9.-]*\.ngrok\.app$|"
            r"http://[a-zA-Z0-9][a-zA-Z0-9.-]*\.ngrok-free\.app(?::\d+)?$"
        ),
        description="Extra allowed origins (regex). Matches public ngrok hostnames. Disable in production with CORS_ALLOW_ORIGIN_REGEX=",
    )

    # File Upload
    max_upload_size: int = Field(default=52428800, description="Max upload size in bytes (50MB)", ge=1024)
    allowed_extensions: List[str] = Field(
        default=[".pdf", ".png", ".jpg", ".jpeg", ".gif", ".tiff", ".tif", ".bmp", ".webp"],
        description="Allowed file extensions",
    )
    upload_dir: str = Field(default="./uploads", description="Upload directory for local storage")

    # Storage
    storage_backend: StorageBackend = Field(default=StorageBackend.LOCAL, description="Storage backend (local or s3)")
    s3_bucket: Optional[str] = Field(default=None, description="AWS S3 bucket name")
    s3_region: Optional[str] = Field(default="us-east-1", description="AWS S3 region")
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS access key ID")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWS secret access key")

    # MongoDB
    mongodb_url: str = Field(default="mongodb://localhost:27017", description="MongoDB connection URL")
    mongodb_username: Optional[str] = Field(default=None, description="MongoDB username")
    mongodb_password: Optional[str] = Field(default=None, description="MongoDB password")
    mongodb_db_name: str = Field(default="dafaai", description="MongoDB database name")
    mongodb_max_pool_size: int = Field(default=10, description="MongoDB max connection pool size", ge=1, le=100)
    mongodb_min_pool_size: int = Field(default=1, description="MongoDB min connection pool size", ge=1, le=10)

    # OCR
    tesseract_path: Optional[str] = Field(default=None, description="Path to Tesseract binary (auto-detect if None)")
    ocr_languages: List[str] = Field(default=["eng"], description="OCR languages (Tesseract language codes)")
    ocr_dpi: int = Field(default=300, description="OCR DPI for image conversion", ge=150, le=600)
    ocr_provider: str = Field(default="google", description="Default OCR provider (google, aws, azure)")

    # Security - API Keys (legacy, kept for backward compat)
    api_keys: List[str] = Field(default=["1234"], description="Valid API keys for authentication")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")

    # Security - JWT
    jwt_secret: str = Field(
        default="change-me-in-production-minimum-32-characters-long",
        description="Secret key for JWT signing",
        min_length=32,
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiry in minutes", ge=1)
    refresh_token_expire_days: int = Field(default=7, description="Refresh token expiry in days", ge=1)

    # Google OAuth
    google_client_id: str = Field(default="", description="Google OAuth Client ID")
    google_client_secret: str = Field(default="", description="Google OAuth Client Secret")

    # Email (Zoho SMTP)
    smtp_host: str = Field(default="smtp.zoho.com", description="SMTP server host")
    smtp_port: int = Field(default=587, description="SMTP server port (587 for TLS)")
    smtp_username: str = Field(default="", description="SMTP login email (e.g. auth@merodafa.com)")
    smtp_password: str = Field(default="", description="SMTP app password")
    smtp_from_email: str = Field(default="auth@merodafa.com", description="From email address")
    smtp_from_name: str = Field(default="MeroDafa", description="From display name")
    verification_code_expire_minutes: int = Field(default=10, description="OTP code expiry in minutes")

    # Frontend URL (used in invite emails, password reset links, etc.)
    frontend_url: str = Field(
        default="http://localhost:5173",
        description="Public URL of the frontend app — used to build invite / reset links.",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    # Redis (optional, for future use)
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL")

    # OpenAI (for embeddings)
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key for embeddings")
    openai_embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")

    # Gemini (for LLM)
    gemini_api_key: Optional[str] = Field(default=None, description="Gemini API key for LLM calls")
    gemini_model: str = Field(default="gemini-2.5-flash", description="Gemini model name")

    # Qdrant (Vector DB)
    qdrant_url: str = Field(default="http://localhost", description="Qdrant server URL")
    qdrant_port: int = Field(default=6333, description="Qdrant server port", ge=1, le=65535)
    qdrant_collection_name: str = Field(default="finance_acts", description="Qdrant collection name")

    # Hybrid Search
    hybrid_search_enabled: bool = Field(default=True, description="Enable hybrid search (vector + BM25)")
    hybrid_search_vector_weight: float = Field(
        default=0.7, description="Weight for vector search in hybrid (0.0-1.0)", ge=0.0, le=1.0
    )
    hybrid_search_bm25_weight: float = Field(
        default=0.3, description="Weight for BM25 search in hybrid (0.0-1.0)", ge=0.0, le=1.0
    )

    # Re-ranking
    rerank_enabled: bool = Field(default=True, description="Enable re-ranking of retrieved chunks")
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for re-ranking",
    )
    rerank_top_k: int = Field(default=5, description="Number of top chunks to return after re-ranking", ge=1, le=20)
    rerank_retrieve_k: int = Field(
        default=20, description="Number of chunks to retrieve before re-ranking", ge=1, le=50
    )
    # Metadata Enhancement
    metadata_enhancement_enabled: bool = Field(
        default=True, description="Enable metadata-based filtering and boosting for retrieval"
    )
    metadata_boost_weight: float = Field(
        default=0.2, description="Weight for metadata-based score boosting (0.0-1.0)", ge=0.0, le=1.0
    )

    # Context Window Optimization
    context_optimization_enabled: bool = Field(
        default=True, description="Enable context window optimization for LLM calls"
    )
    max_context_tokens: int = Field(
        default=30000, description="Maximum context tokens for LLM (Gemini 2.5 Flash: 30000)", ge=1000, le=100000
    )
    context_reserved_tokens: int = Field(
        default=2000, description="Reserved tokens for prompt/system/query", ge=500, le=5000
    )

    # Feature Flags
    enable_metrics: bool = Field(default=False, description="Enable metrics collection")
    enable_tracing: bool = Field(default=False, description="Enable distributed tracing")
    default_retrieval_strategy: RetrievalStrategyType = Field(
        default=RetrievalStrategyType.PAGE_INDEX,
        description="Default retrieval strategy: 'page_index' (vectorless RAG) or 'vector' (Qdrant+embeddings). "
        "Can be overridden per-request.",
    )

    # Download Retry
    download_retry_attempts: int = Field(
        default=3, description="Number of retry attempts for URL downloads", ge=1, le=10
    )
    download_retry_wait_min: float = Field(default=1.0, description="Minimum wait between retries in seconds", ge=0.1)
    download_retry_wait_max: float = Field(default=10.0, description="Maximum wait between retries in seconds", ge=1.0)

    # Batch document injection (cron / CLI)
    injection_sources_config: str = Field(
        default="config/sources.yaml",
        description="Path to YAML source definitions (relative to backend root or absolute)",
    )
    injection_concurrency: int = Field(
        default=2, description="Max concurrent documents processed during injection", ge=1, le=16
    )
    injection_retry_attempts: int = Field(
        default=3, description="Reserved: per-document retries (sources use httpx defaults)", ge=1, le=10
    )
    injection_scraper_delay: float = Field(
        default=1.0, description="Delay between HTTP requests when scraping sources (seconds)", ge=0.0, le=60.0
    )
    injection_local_log_path: str = Field(
        default="data/injection_log.json",
        description="JSON file for injection dedupe/history when MongoDB is unavailable (relative to backend root or absolute)",
    )
    injection_download_dir: str = Field(
        default="data/downloaded",
        description="Default base directory for --download-only (relative to backend root or absolute)",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("cors_allow_origin_regex", mode="before")
    @classmethod
    def parse_cors_allow_origin_regex(cls, v):
        """Unset empty string so production can disable regex with CORS_ALLOW_ORIGIN_REGEX= """
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def parse_allowed_extensions(cls, v):
        """Parse allowed extensions from string or list."""
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v

    @field_validator("ocr_languages", mode="before")
    @classmethod
    def parse_ocr_languages(cls, v):
        """Parse OCR languages from string or list."""
        if isinstance(v, str):
            return [lang.strip() for lang in v.split(",")]
        return v

    @field_validator("api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, v):
        """Parse API keys from string or list."""
        if isinstance(v, str):
            return [key.strip() for key in v.split(",")]
        return v

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == Environment.PRODUCTION

    def validate_s3_config(self) -> bool:
        """Validate S3 configuration if S3 backend is enabled."""
        if self.storage_backend == StorageBackend.S3:
            return all(
                [
                    self.s3_bucket,
                    self.s3_region,
                    self.aws_access_key_id,
                    self.aws_secret_access_key,
                ]
            )
        return True


# Global settings instance
settings = Settings()

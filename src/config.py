"""
Centralised settings — reads from .env / environment variables.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Anthropic
    anthropic_api_key: str = ""

    # Voyage AI (embeddings) — get free key at https://dash.voyageai.com/
    # Falls back to anthropic_api_key if not set (for Anthropic customers)
    voyage_api_key: str = ""

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j"
    neo4j_database: str = "neo4j"

    # App
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "info"

    # Linking thresholds
    # Calibrated for Voyage AI voyage-3 document-to-document cosine similarity.
    # voyage-3 doc↔doc scores range ~0.08–0.55; 0.38 keeps only genuinely related captures.
    cosine_similarity_threshold: float = 0.65
    temporal_proximity_days: int = 90
    link_strength_min: float = 0.25


settings = Settings()

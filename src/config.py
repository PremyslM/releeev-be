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

    # Auth — set a strong JWT_SECRET in .env for production.
    jwt_secret: str = "dev-secret-change-me"
    jwt_ttl_days: int = 30

    # Run the idempotent schema bootstrap on startup. Safe to leave on locally;
    # on serverless (Vercel) set RUN_SCHEMA_SETUP=false after creating the schema
    # once, so cold starts don't re-run it.
    run_schema_setup: bool = True

    # Linking thresholds
    # Calibrated for Voyage AI voyage-3 document-to-document cosine similarity.
    # voyage-3 doc↔doc scores range ~0.08–0.55; 0.38 keeps only genuinely related captures.
    cosine_similarity_threshold: float = 0.65
    temporal_proximity_days: int = 90
    link_strength_min: float = 0.25

    # ── Environment helpers ──────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.app_env.strip().lower() in ("production", "prod")

    def safety_problems(self) -> tuple[list[str], list[str]]:
        """
        Validate the config for the current environment.

        Returns (fatal, warnings). `fatal` items are misconfigurations that make
        a *production* deploy insecure or non-functional — the app should refuse
        to boot. `warnings` are logged but non-blocking. In non-production
        environments nothing is fatal (local dev / tests stay frictionless).
        """
        fatal: list[str] = []
        warnings: list[str] = []
        if self.is_production:
            if self.jwt_secret == "dev-secret-change-me" or len(self.jwt_secret) < 16:
                fatal.append("JWT_SECRET must be a strong random value (≥16 chars) in production.")
            if not self.anthropic_api_key:
                warnings.append("ANTHROPIC_API_KEY is empty — capture analysis will fail.")
            if self.neo4j_password in ("", "neo4j", "changeme", "12345678"):
                warnings.append("NEO4J_PASSWORD looks like a placeholder/default.")
        return fatal, warnings


settings = Settings()

"""
Centralised settings — reads from .env / environment variables.
"""
from pydantic import AliasChoices, Field
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
    # Model routing to control token cost. `claude_model` (capable) is used only
    # where quality matters — image understanding, the contextual insight, and the
    # RAG answer. Everything else (extraction, expansion, themes, journal) uses the
    # much cheaper `claude_model_fast`.
    claude_model: str = "claude-sonnet-5"
    claude_model_fast: str = "claude-haiku-4-5-20251001"

    # Voyage AI (embeddings) — get free key at https://dash.voyageai.com/
    # Falls back to anthropic_api_key if not set (for Anthropic customers)
    voyage_api_key: str = ""

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    # Aura's downloadable credentials file says NEO4J_USERNAME; accept both, so
    # pasting that file into Vercel can't silently fall back to "neo4j" (AuthError).
    neo4j_user: str = Field("neo4j", validation_alias=AliasChoices("NEO4J_USER", "NEO4J_USERNAME"))
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

    # Bug reporting → GitHub Issues. Server-side token (never shipped in the app).
    # GITHUB_TOKEN needs `repo` scope; GITHUB_REPO is "owner/name".
    github_token: str = ""
    github_repo: str = ""
    # Public base URL of this backend (e.g. https://releeev-be.vercel.app), used to
    # build links GitHub can fetch (bug screenshots). Falls back to the request host.
    public_base_url: str = ""

    # Run the idempotent schema bootstrap on startup. Safe to leave on locally;
    # on serverless (Vercel) set RUN_SCHEMA_SETUP=false after creating the schema
    # once, so cold starts don't re-run it.
    run_schema_setup: bool = True

    # Shared secret Vercel Cron sends as "Authorization: Bearer <CRON_SECRET>".
    # Guards /cron/keepalive, the daily write that keeps an Aura Free instance
    # from being paused (and, after a long pause, deleted).
    cron_secret: str = ""

    # Search reranking — a Voyage cross-encoder reorders the fused candidate set
    # for sharper top results. Falls back to the RRF order on any error.
    rerank_enabled: bool = True
    rerank_model: str = "rerank-2-lite"   # fast + cheap; "rerank-2" is stronger
    rerank_candidates: int = 30           # how many top RRF hits to rerank
    # Drop results whose reranker relevance is below this — keeps search precise
    # (only genuinely-related hits) instead of returning every fused match.
    rerank_min_score: float = 0.35

    # Linking thresholds
    # Calibrated for Voyage AI voyage-3 document-to-document cosine similarity.
    # voyage-3 doc↔doc scores range ~0.08–0.55; 0.38 keeps only genuinely related captures.
    cosine_similarity_threshold: float = 0.65
    temporal_proximity_days: int = 90
    link_strength_min: float = 0.25
    # Cap on how many links a single capture keeps (its strongest) — keeps the
    # graph sparse and readable instead of a fully-connected hairball.
    max_links_per_capture: int = 6

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
            if not self.cron_secret:
                warnings.append("CRON_SECRET is empty — the keepalive cron is refused, so an "
                                "idle Aura Free instance can be paused and later deleted.")
        return fatal, warnings


settings = Settings()

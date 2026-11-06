"""Runtime configuration. Everything comes from environment variables / .env; secrets never live in code."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Which model runs the agents. Format: "<provider>:<model>", e.g.
    #   anthropic:claude-sonnet-4-5   openai:gpt-4.1-mini   ollama:qwen3:8b   test:test
    llm_model: str = Field(default="ollama:qwen3:8b", alias="LLM_MODEL")
    ollama_base_url: str = Field(default="http://127.0.0.1:11434/v1", alias="OLLAMA_BASE_URL")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    # Storage. Unset DATABASE_URL -> SQLite file + in-memory vector index (dev/tests);
    # postgresql://... -> PostgreSQL with pgvector (docker compose).
    database_url: str = Field(default="sqlite:///./credit_ops.db", alias="DATABASE_URL")
    # fastembed model id; "hash" = dependency-free bag-of-words embedder for unit tests / CI
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", alias="EMBEDDING_MODEL"
    )
    policy_dir: str = Field(default="app/policy", alias="POLICY_DIR")

    # Guardrails
    max_agent_retries: int = Field(default=2, alias="MAX_AGENT_RETRIES")
    auto_approve_max_amount_uah: int = Field(default=300_000, alias="AUTO_APPROVE_MAX_AMOUNT_UAH")
    crm_webhook_url: str | None = Field(default=None, alias="CRM_WEBHOOK_URL")
    log_json: bool = Field(default=False, alias="LOG_JSON")


settings = Settings()

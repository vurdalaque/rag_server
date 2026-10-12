"""Environment-backed configuration for retrieval observability and evaluation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from radius_rag.llm.params import env_bool, env_float, env_int

_RAG_SERVER_ROOT = Path(__file__).resolve().parents[3]
MIGRATIONS_DIR = Path(
    os.getenv("RETRIEVAL_MIGRATIONS_DIR", str(_RAG_SERVER_ROOT / "retrieval_migrations"))
)

# Feature flags
TELEMETRY_ENABLED = env_bool("RETRIEVAL_TELEMETRY_ENABLED", False)
EVALUATION_AUTO_ENQUEUE = env_bool("RETRIEVAL_EVALUATION_AUTO_ENQUEUE", False)

DATABASE_URL = os.getenv("RETRIEVAL_DATABASE_URL", "").strip()

# Telemetry writer
TELEMETRY_QUEUE_SIZE = max(1, env_int("RETRIEVAL_TELEMETRY_QUEUE_SIZE", 1000))
TELEMETRY_BATCH_SIZE = max(1, env_int("RETRIEVAL_TELEMETRY_BATCH_SIZE", 50))
TELEMETRY_FLUSH_INTERVAL_SECONDS = max(
    0.05,
    env_float("RETRIEVAL_TELEMETRY_FLUSH_INTERVAL_SECONDS", 1.0),
)

# Evaluation worker
EVALUATOR_CONCURRENCY = max(1, env_int("RETRIEVAL_EVALUATOR_CONCURRENCY", 1))
EVALUATION_LEASE_TTL_SECONDS = max(5, env_int("RETRIEVAL_EVALUATION_LEASE_TTL_SECONDS", 300))
EVALUATION_MAX_RETRIES = max(0, env_int("RETRIEVAL_EVALUATION_MAX_RETRIES", 3))
EVALUATION_RETRY_BASE_SECONDS = max(
    0.1,
    env_float("RETRIEVAL_EVALUATION_RETRY_BASE_SECONDS", 5.0),
)
EVALUATION_POLL_INTERVAL_SECONDS = max(
    0.1,
    env_float("RETRIEVAL_EVALUATION_POLL_INTERVAL_SECONDS", 2.0),
)

# Judge pipeline (separate from RAG_LLM_*)
JUDGE_PROMPT_VERSION = os.getenv("RETRIEVAL_JUDGE_PROMPT_VERSION", "v1").strip() or "v1"
JUDGE_LLM_URL = os.getenv(
    "RETRIEVAL_JUDGE_LLM_URL",
    os.getenv("LLM_URL", "http://127.0.0.1:8001/v1/chat/completions"),
)
JUDGE_LLM_MODEL = os.getenv("RETRIEVAL_JUDGE_LLM_MODEL", os.getenv("LLM_MODEL", "rag-assistant"))
JUDGE_LLM_TEMPERATURE = env_float("RETRIEVAL_JUDGE_LLM_TEMPERATURE", 0.0)
JUDGE_LLM_TOP_P = env_float("RETRIEVAL_JUDGE_LLM_TOP_P", 1.0)
JUDGE_LLM_TOP_K = env_int("RETRIEVAL_JUDGE_LLM_TOP_K", 20)
JUDGE_LLM_PRESENCE_PENALTY = env_float("RETRIEVAL_JUDGE_LLM_PRESENCE_PENALTY", 0.0)
JUDGE_LLM_REPETITION_PENALTY = env_float("RETRIEVAL_JUDGE_LLM_REPETITION_PENALTY", 1.0)
JUDGE_LLM_ENABLE_THINKING = env_bool("RETRIEVAL_JUDGE_LLM_ENABLE_THINKING", False)
JUDGE_LLM_TIMEOUT_SECONDS = env_float(
    "RETRIEVAL_JUDGE_LLM_TIMEOUT_SECONDS",
    env_float("REQUEST_TIMEOUT", 120.0),
)

# Timescale migration_config key for compression delay (see 005_compression_policies.sql)
COMPRESSION_CONFIG_KEY = os.getenv(
    "RETRIEVAL_COMPRESSION_CONFIG_KEY",
    "retrieval_events_compress_after",
)

# Judge inputs and metric thresholds
JUDGE_RELEVANCE_THRESHOLD = env_int("RETRIEVAL_JUDGE_RELEVANCE_THRESHOLD", 2)
JUDGE_MAX_QUERY_CHARS = max(256, env_int("RETRIEVAL_JUDGE_MAX_QUERY_CHARS", 8000))
JUDGE_MAX_CHUNK_CHARS = max(256, env_int("RETRIEVAL_JUDGE_MAX_CHUNK_CHARS", 6000))
JUDGE_MAX_CHUNKS = max(1, env_int("RETRIEVAL_JUDGE_MAX_CHUNKS", 30))
JUDGE_JSON_MAX_RETRIES = max(0, env_int("RETRIEVAL_JUDGE_JSON_MAX_RETRIES", 2))


@dataclass(frozen=True)
class RetrievalConfig:
    telemetry_enabled: bool
    evaluation_auto_enqueue: bool
    database_url: str
    queue_size: int
    batch_size: int
    flush_interval_seconds: float
    evaluator_concurrency: int
    lease_ttl_seconds: int
    max_retries: int
    retry_base_seconds: float
    poll_interval_seconds: float
    judge_prompt_version: str
    judge_llm_url: str
    judge_llm_model: str
    judge_llm_temperature: float
    judge_llm_top_p: float
    judge_llm_top_k: int
    judge_llm_presence_penalty: float
    judge_llm_repetition_penalty: float
    judge_llm_enable_thinking: bool
    judge_llm_timeout_seconds: float
    compression_config_key: str
    judge_relevance_threshold: int
    judge_max_query_chars: int
    judge_max_chunk_chars: int
    judge_max_chunks: int
    judge_json_max_retries: int
    migrations_dir: Path

    @property
    def evaluation_version(self) -> str:
        from radius_rag.observability.evaluation.identity import build_evaluation_version

        return build_evaluation_version(self.judge_prompt_version, self.judge_llm_model)

    def telemetry_ready(self) -> bool:
        return self.telemetry_enabled and bool(self.database_url)


def load_retrieval_config() -> RetrievalConfig:
    return RetrievalConfig(
        telemetry_enabled=TELEMETRY_ENABLED,
        evaluation_auto_enqueue=EVALUATION_AUTO_ENQUEUE,
        database_url=DATABASE_URL,
        queue_size=TELEMETRY_QUEUE_SIZE,
        batch_size=TELEMETRY_BATCH_SIZE,
        flush_interval_seconds=TELEMETRY_FLUSH_INTERVAL_SECONDS,
        evaluator_concurrency=EVALUATOR_CONCURRENCY,
        lease_ttl_seconds=EVALUATION_LEASE_TTL_SECONDS,
        max_retries=EVALUATION_MAX_RETRIES,
        retry_base_seconds=EVALUATION_RETRY_BASE_SECONDS,
        poll_interval_seconds=EVALUATION_POLL_INTERVAL_SECONDS,
        judge_prompt_version=JUDGE_PROMPT_VERSION,
        judge_llm_url=JUDGE_LLM_URL,
        judge_llm_model=JUDGE_LLM_MODEL,
        judge_llm_temperature=JUDGE_LLM_TEMPERATURE,
        judge_llm_top_p=JUDGE_LLM_TOP_P,
        judge_llm_top_k=JUDGE_LLM_TOP_K,
        judge_llm_presence_penalty=JUDGE_LLM_PRESENCE_PENALTY,
        judge_llm_repetition_penalty=JUDGE_LLM_REPETITION_PENALTY,
        judge_llm_enable_thinking=JUDGE_LLM_ENABLE_THINKING,
        judge_llm_timeout_seconds=JUDGE_LLM_TIMEOUT_SECONDS,
        compression_config_key=COMPRESSION_CONFIG_KEY,
        judge_relevance_threshold=JUDGE_RELEVANCE_THRESHOLD,
        judge_max_query_chars=JUDGE_MAX_QUERY_CHARS,
        judge_max_chunk_chars=JUDGE_MAX_CHUNK_CHARS,
        judge_max_chunks=JUDGE_MAX_CHUNKS,
        judge_json_max_retries=JUDGE_JSON_MAX_RETRIES,
        migrations_dir=MIGRATIONS_DIR,
    )

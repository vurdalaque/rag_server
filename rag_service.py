#!/usr/bin/env python3

import json
import math

import os
import re
import time

from pathlib import Path
from typing import Any

import faiss
import httpx
import numpy as np

from llm_params import apply_llm_defaults
from rag_metrics import (
    BM25_DURATION,
    FAISS_DURATION,
    RERANK_DURATION,
    record_context_chars,
    record_retrieve_empty,
    record_retrieve_result,
    track_dependency,
    track_duration,
    update_index_state,
)


INDEX_FILE = Path(
    os.getenv(
        "RAG_INDEX_FILE",
        "rag_index.faiss",
    )
)

METADATA_FILE = Path(
    os.getenv(
        "RAG_METADATA_FILE",
        "rag_metadata.jsonl",
    )
)


LLM_URL = os.getenv(
    "LLM_URL",
    "http://127.0.0.1:8001/v1/chat/completions",
)

EMBEDDING_URL = os.getenv(
    "EMBEDDING_URL",
    "http://127.0.0.1:8002/v1/embeddings",
)

RERANK_URL = os.getenv(
    "RERANK_URL",
    "http://127.0.0.1:8002/v1/rerank",
)


LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "rag-assistant",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "qwen3-embedding",
)

# Обычно совпадает с именем модели,
# опубликованным embedding-сервером.
RERANK_MODEL = os.getenv(
    "RERANK_MODEL",
    EMBEDDING_MODEL,
)


DEFAULT_TOP_K = int(
    os.getenv(
        "RAG_TOP_K",
        "6",
    )
)

MAX_TOP_K = int(
    os.getenv(
        "RAG_MAX_TOP_K",
        "20",
    )
)

# Сколько кандидатов FAISS передавать reranker.
RERANK_CANDIDATES = int(
    os.getenv(
        "RAG_RERANK_CANDIDATES",
        "30",
    )
)

# Максимальная длина одного документа,
# передаваемого reranker.
RERANK_DOCUMENT_MAX_CHARS = int(
    os.getenv(
        "RAG_RERANK_DOCUMENT_MAX_CHARS",
        "12000",
    )
)

MAX_CONTEXT_CHARS = int(
    os.getenv(
        "RAG_MAX_CONTEXT_CHARS",
        "24000",
    )
)

REQUEST_TIMEOUT = float(
    os.getenv(
        "REQUEST_TIMEOUT",
        "600",
    )
)


def env_bool(
    name: str,
    default: bool,
) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


RERANK_ENABLED = env_bool(
    "RAG_RERANK_ENABLED",
    False,
)

# При ошибке reranker вернуть результаты FAISS,
# а не завершать весь поиск ошибкой.
RERANK_FALLBACK = env_bool(
    "RAG_RERANK_FALLBACK",
    True,
)

# Минимальная cosine similarity на первом этапе.
# Значение -1 фактически отключает фильтрацию.
MIN_FAISS_SCORE = float(
    os.getenv(
        "RAG_MIN_FAISS_SCORE",
        "-1.0",
    )
)

# Фильтр relevance_score выключен по умолчанию,
# потому что диапазон оценок зависит от модели.
MIN_RERANK_SCORE_ENV = os.getenv(
    "RAG_MIN_RERANK_SCORE"
)

MIN_RERANK_SCORE = (
    float(MIN_RERANK_SCORE_ENV)
    if MIN_RERANK_SCORE_ENV is not None
    else None
)


def source_type_boosts() -> dict[str, float]:
    raw_value = os.getenv("RAG_SOURCE_TYPE_BOOSTS", "{}")

    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise RuntimeError("RAG_SOURCE_TYPE_BOOSTS must be a JSON object") from error

    if not isinstance(value, dict):
        raise RuntimeError("RAG_SOURCE_TYPE_BOOSTS must be a JSON object")

    boosts: dict[str, float] = {}

    for source_type, boost in value.items():
        if not isinstance(source_type, str):
            raise RuntimeError("RAG_SOURCE_TYPE_BOOSTS keys must be strings")

        try:
            boosts[source_type] = float(boost)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "RAG_SOURCE_TYPE_BOOSTS values must be numbers"
            ) from error

    return boosts


SOURCE_TYPE_BOOSTS = source_type_boosts()
BM25_K1 = 1.5
BM25_B = 0.75



class RagService:
    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []
        self._bm25_terms: list[dict[str, int]] = []
        self._bm25_document_frequency: dict[str, int] = {}
        self._bm25_average_length = 0.0

    def load(self) -> None:
        self.load_from_paths(
            Path(os.getenv("RAG_INDEX_FILE", "rag_index.faiss")),
            Path(os.getenv("RAG_METADATA_FILE", "rag_metadata.jsonl")),
        )

    def load_if_available(self) -> bool:
        index_file = Path(os.getenv("RAG_INDEX_FILE", "rag_index.faiss"))
        metadata_file = Path(os.getenv("RAG_METADATA_FILE", "rag_metadata.jsonl"))

        if not index_file.exists() or not metadata_file.exists():
            self.clear()
            return False

        self.load_from_paths(index_file, metadata_file)
        return True

    def clear(self) -> None:
        self.index = None
        self.metadata = []
        self._bm25_terms = []
        self._bm25_document_frequency = {}
        self._bm25_average_length = 0.0
        update_index_state(self)

    def load_from_paths(self, index_file: Path, metadata_file: Path) -> None:
        if not index_file.exists():
            raise RuntimeError(f"FAISS index not found: {index_file}")

        if not metadata_file.exists():
            raise RuntimeError(f"Metadata file not found: {metadata_file}")

        index = faiss.read_index(str(index_file))
        metadata = self._load_metadata(metadata_file)

        if index.ntotal != len(metadata):
            raise RuntimeError(f"Index/metadata mismatch: {index.ntotal} != {len(metadata)}")

        self.index = index
        self.metadata = metadata
        self._build_bm25_index()
        update_index_state(self)

    @staticmethod
    def _load_metadata(
        path: Path,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        with path.open(
            "r",

            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        "Invalid metadata JSON at line "
                        f"{line_number}: {error}"
                    ) from error

                if not isinstance(item, dict):
                    raise RuntimeError(
                        "Invalid metadata object at line "
                        f"{line_number}"
                    )

                result.append(item)

        return result

    @property
    def index_loaded(self) -> bool:
        return self.index is not None

    @property
    def document_count(self) -> int:
        if self.index is None:
            return 0

        return int(self.index.ntotal)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())

    def _build_bm25_index(self) -> None:
        document_frequency: dict[str, int] = {}
        term_counts: list[dict[str, int]] = []
        total_length = 0

        for item in self.metadata:
            text = " ".join(
                str(item.get(field, ""))
                for field in ("file", "language", "full_name", "name", "detail", "code")
            )
            counts: dict[str, int] = {}

            for token in self._tokens(text):
                counts[token] = counts.get(token, 0) + 1

            total_length += sum(counts.values())
            term_counts.append(counts)

            for token in counts:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        self._bm25_terms = term_counts
        self._bm25_document_frequency = document_frequency
        self._bm25_average_length = total_length / len(term_counts) if term_counts else 0.0

    @staticmethod
    def _normalize_path(path: str) -> str:
        return path.replace("\\", "/")

    @staticmethod
    def _path_matches(file_name: str, path_prefix: str) -> bool:
        normalized_file = RagService._normalize_path(file_name).lower()
        prefix = RagService._normalize_path(path_prefix).strip("/").lower()

        if not prefix:
            return True

        if normalized_file.startswith(prefix):
            return True

        if normalized_file.startswith(f"/{prefix}"):
            return True

        file_segments = [segment for segment in normalized_file.split("/") if segment]
        prefix_segments = [segment for segment in prefix.split("/") if segment]

        if not prefix_segments:
            return True

        for index in range(len(file_segments) - len(prefix_segments) + 1):
            if file_segments[index:index + len(prefix_segments)] == prefix_segments:
                return True

        return False

    @staticmethod
    def _matches_filters(
        item: dict[str, Any],
        source_type: str | None,
        language: str | None,
        path_prefix: str | None,
    ) -> bool:
        if source_type is not None and item.get("source_type", "code") != source_type:
            return False

        if language is not None and item.get("language", "") != language:
            return False

        if path_prefix is not None:
            file_name = str(item.get("file", ""))
            if not RagService._path_matches(file_name, path_prefix):
                return False

        return True

    def _bm25_scores(
        self,
        query: str,
        allowed_indices: set[int],
    ) -> dict[int, float]:
        tokens = self._tokens(query)
        if not tokens or not self._bm25_average_length:
            return {}

        document_count = len(self._bm25_terms)
        scores: dict[int, float] = {}

        for item_index in allowed_indices:
            counts = self._bm25_terms[item_index]
            document_length = sum(counts.values())
            score = 0.0

            for token in tokens:
                frequency = counts.get(token, 0)
                if not frequency:
                    continue

                document_frequency = self._bm25_document_frequency.get(token, 0)
                inverse_frequency = max(
                    0.0,
                    math.log(
                        (document_count - document_frequency + 0.5)
                        / (document_frequency + 0.5)
                        + 1.0
                    ),
                )
                denominator = frequency + BM25_K1 * (
                    1.0 - BM25_B + BM25_B * document_length / self._bm25_average_length
                )
                score += inverse_frequency * frequency * (BM25_K1 + 1.0) / denominator

            if score:
                scores[item_index] = score

        return scores

    @property
    def dimensions(self) -> int:
        if self.index is None:
            return 0

        return int(self.index.d)

    @property
    def rerank_enabled(self) -> bool:
        return RERANK_ENABLED

    @staticmethod
    def _normalize_top_k(
        top_k: int | None,
    ) -> int:
        value = (
            top_k
            if top_k is not None
            else DEFAULT_TOP_K
        )

        return max(
            1,
            min(
                int(value),
                MAX_TOP_K,
            ),
        )

    def _candidate_count(
        self,
        top_k: int,
    ) -> int:
        if self.index is None:
            raise RuntimeError(
                "FAISS index is not loaded"
            )

        requested = max(
            top_k,
            RERANK_CANDIDATES,
        )

        return min(
            requested,
            int(self.index.ntotal),
        )

    @staticmethod
    def _metadata_to_result(
        item: dict[str, Any],
        item_index: int,
        faiss_score: float,
    ) -> dict[str, Any]:
        return {
            "metadata_index": item_index,
            "score": faiss_score,
            "faiss_score": faiss_score,
            "rerank_score": None,
            "source_type": item.get("source_type", "code"),

            "file": item.get("file", ""),
            "language": item.get(
                "language",
                "",
            ),
            "symbol": (
                item.get("full_name")
                or item.get("name")
                or ""
            ),
            "signature": item.get(
                "detail",
                "",
            ),
            "start_line": item.get(
                "start_line"
            ),
            "end_line": item.get(
                "end_line"
            ),
            "code": item.get(
                "code",
                "",
            ),
        }

    @staticmethod
    def _build_rerank_document(
        candidate: dict[str, Any],
    ) -> str:
        document = "\n".join(
            [
                (
                    "Symbol: "
                    f"{candidate.get('symbol', '')}"
                ),
                (
                    "Signature: "
                    f"{candidate.get('signature', '')}"
                ),
                (
                    "Language: "
                    f"{candidate.get('language', '')}"
                ),
                (
                    "Source type: "
                    f"{candidate.get('source_type', 'code')}"
                ),

                (
                    "File: "
                    f"{candidate.get('file', '')}"
                ),
                (
                    "Lines: "
                    f"{candidate.get('start_line')}-"
                    f"{candidate.get('end_line')}"
                ),
                "",
                "Source code:",
                str(
                    candidate.get(
                        "code",
                        "",
                    )
                ),
            ]
        )

        if len(document) > RERANK_DOCUMENT_MAX_CHARS:
            document = document[
                :RERANK_DOCUMENT_MAX_CHARS
            ]

        return document

    async def get_embedding(
        self,
        client: httpx.AsyncClient,
        text: str,
    ) -> np.ndarray:
        with track_dependency("embedding"):
            response = await client.post(
                EMBEDDING_URL,
                json={
                    "model": EMBEDDING_MODEL,
                    "input": [text],
                },
            )

            response.raise_for_status()

        payload = response.json()
        data = payload.get("data")

        if (
            not isinstance(data, list)
            or len(data) != 1
        ):
            raise RuntimeError(
                "Unexpected embeddings response"
            )

        embedding = data[0].get(
            "embedding"
        )

        if not isinstance(embedding, list):
            raise RuntimeError(
                "Embedding response has no vector"
            )

        vector = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(1, -1)

        faiss.normalize_L2(vector)

        if self.index is None:
            raise RuntimeError(
                "FAISS index is not loaded"
            )

        if vector.shape[1] != self.index.d:
            raise RuntimeError(
                "Embedding dimensions mismatch: "
                f"{vector.shape[1]} "
                f"!= {self.index.d}"
            )

        return vector

    async def rerank(
        self,
        client: httpx.AsyncClient,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        documents = [
            self._build_rerank_document(
                candidate
            )
            for candidate in candidates
        ]

        with track_dependency("rerank"):
            with track_duration(RERANK_DURATION):
                response = await client.post(
                    RERANK_URL,
                    json={
                        "model": RERANK_MODEL,
                        "query": query,
                        "documents": documents,
                        "top_n": min(
                            top_k,
                            len(documents),
                        ),
                    },
                )

                response.raise_for_status()

        payload = response.json()
        raw_results = payload.get("results")

        if not isinstance(raw_results, list):
            raise RuntimeError(
                "Unexpected rerank response: "
                "'results' is not an array"
            )

        ranked: list[dict[str, Any]] = []

        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                continue

            original_index = raw_result.get(
                "index"
            )

            relevance_score = raw_result.get(
                "relevance_score"
            )

            if not isinstance(original_index, int):
                continue

            if (
                original_index < 0
                or original_index >= len(candidates)
            ):
                continue

            try:
                score = float(
                    relevance_score
                )
            except (TypeError, ValueError):
                continue

            if (
                MIN_RERANK_SCORE is not None
                and score < MIN_RERANK_SCORE
            ):
                continue

            candidate = dict(
                candidates[original_index]
            )

            candidate["rerank_score"] = score

            # Общее поле score теперь означает
            # итоговую оценку после rerank.
            candidate["score"] = score

            ranked.append(candidate)

        return ranked[:top_k]

    @staticmethod
    def _hybrid_fallback(
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        results = [dict(candidate) for candidate in candidates[:top_k]]
        for result in results:
            result["score"] = result["hybrid_score"]
        return results

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_type: str | None = None,
        language: str | None = None,
        path_prefix: str | None = None,
    ) -> list[dict[str, Any]]:
        start = time.perf_counter()
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")
        if self.index is None:
            record_retrieve_empty("index_missing")
            record_retrieve_result(0, False, time.perf_counter() - start)
            return []
        if any(value is not None and not isinstance(value, str) for value in (source_type, language, path_prefix)):
            raise ValueError("RAG filters must be strings")

        limit = self._normalize_top_k(top_k)
        filters_active = any(value is not None for value in (source_type, language, path_prefix))
        candidate_count = self._candidate_count(limit) if RERANK_ENABLED else limit
        allowed_indices = {
            item_index for item_index, item in enumerate(self.metadata)
            if self._matches_filters(item, source_type, language, path_prefix)
        }
        if not allowed_indices:
            record_retrieve_empty("filtered")
            record_retrieve_result(0, filters_active, time.perf_counter() - start)
            return []

        search_count = len(self.metadata) if filters_active else candidate_count
        async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT)) as client:
            vector = await self.get_embedding(client, query)
            with track_duration(FAISS_DURATION):
                scores, indices = self.index.search(vector, search_count)
            faiss_scores: dict[int, float] = {}
            faiss_ranks: dict[int, int] = {}
            for score, item_index in zip(scores[0], indices[0]):
                item_index = int(item_index)
                faiss_score = float(score)
                if item_index not in allowed_indices or faiss_score < MIN_FAISS_SCORE:
                    continue
                faiss_scores[item_index] = faiss_score
                faiss_ranks[item_index] = len(faiss_ranks) + 1
                if len(faiss_ranks) >= candidate_count:
                    break

            with track_duration(BM25_DURATION):
                bm25_scores = self._bm25_scores(query, allowed_indices)
            bm25_ranked = sorted(bm25_scores, key=bm25_scores.get, reverse=True)[:candidate_count]
            bm25_ranks = {item_index: rank for rank, item_index in enumerate(bm25_ranked, start=1)}
            candidate_indices = set(faiss_ranks) | set(bm25_ranks)
            if not candidate_indices:
                record_retrieve_empty("no_candidates")
                record_retrieve_result(0, filters_active, time.perf_counter() - start)
                return []

            candidates: list[dict[str, Any]] = []
            for item_index in candidate_indices:
                faiss_score = faiss_scores.get(item_index)
                candidate = self._metadata_to_result(
                    self.metadata[item_index], item_index, faiss_score or 0.0
                )
                candidate["faiss_score"] = faiss_score
                candidate["bm25_score"] = bm25_scores.get(item_index)
                rank_score = sum(
                    1.0 / (60 + rank)
                    for rank in (faiss_ranks.get(item_index), bm25_ranks.get(item_index))
                    if rank is not None
                )
                candidate["hybrid_score"] = rank_score * SOURCE_TYPE_BOOSTS.get(
                    str(candidate["source_type"]), 1.0
                )
                candidate["score"] = candidate["hybrid_score"]
                candidates.append(candidate)

            candidates.sort(key=lambda candidate: candidate["hybrid_score"], reverse=True)
            if not RERANK_ENABLED:
                results = self._hybrid_fallback(candidates, limit)
                record_retrieve_result(
                    len(results),
                    filters_active,
                    time.perf_counter() - start,
                )
                return results

            try:
                reranked = await self.rerank(client, query, candidates, limit)
                if reranked:
                    for candidate in reranked:
                        candidate["score"] = candidate["rerank_score"] * SOURCE_TYPE_BOOSTS.get(
                            str(candidate["source_type"]), 1.0
                        )
                    results = sorted(reranked, key=lambda candidate: candidate["score"], reverse=True)
                    record_retrieve_result(
                        len(results),
                        filters_active,
                        time.perf_counter() - start,
                    )
                    return results
                if not RERANK_FALLBACK:
                    record_retrieve_result(0, filters_active, time.perf_counter() - start)
                    return []
                print("Reranker returned no results; falling back to hybrid ranking")
            except (httpx.HTTPError, RuntimeError, ValueError) as error:
                if not RERANK_FALLBACK:
                    raise
                print(f"Rerank failed; falling back to hybrid: {error}")

            results = self._hybrid_fallback(candidates, limit)
            record_retrieve_result(
                len(results),
                filters_active,
                time.perf_counter() - start,
            )
            return results

    @staticmethod
    def build_context(
        results: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        chunks: list[str] = []
        sources: list[dict[str, Any]] = []
        total_chars = 0

        for position, result in enumerate(results, start=1):
            score_lines: list[str] = []
            if result.get("rerank_score") is not None:
                score_lines.append(f"Rerank relevance: {float(result['rerank_score']):.6f}")
            if result.get("faiss_score") is not None:
                score_lines.append(f"FAISS similarity: {float(result['faiss_score']):.6f}")
            if result.get("bm25_score") is not None:
                score_lines.append(f"BM25 score: {float(result['bm25_score']):.6f}")

            chunk = "\n".join([
                f"[Source {position}]",
                *score_lines,
                f"Source type: {result.get('source_type', 'code')}",
                f"Language: {result['language']}",
                f"File: {result['file']}",
                f"Symbol: {result['symbol']}",
                f"Signature: {result['signature']}",
                f"Lines: {result['start_line']}-{result['end_line']}",
                "Code:",
                str(result["code"]),
            ])
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk[:remaining]
            chunks.append(chunk)
            total_chars += len(chunk)
            sources.append({
                "score": result["score"],
                "rerank_score": result.get("rerank_score"),
                "faiss_score": result.get("faiss_score"),
                "file": result["file"],
                "source_type": result.get("source_type", "code"),
                "symbol": result["symbol"],
                "start_line": result["start_line"],
                "end_line": result["end_line"],
            })

        context = "\n\n---\n\n".join(chunks)
        record_context_chars(len(context))
        return context, sources

    @staticmethod
    def enrich_messages(
        original_messages: list[
            dict[str, Any]
        ],
        context: str,
        use_system_prompt: bool = True,
    ) -> list[dict[str, Any]]:
        system_parts: list[str] = []
        other_messages: list[
            dict[str, Any]
        ] = []

        for message in original_messages:
            if message.get("role") == "system":
                content = message.get(
                    "content"
                )

                if isinstance(content, str):
                    system_parts.append(content)
            else:
                other_messages.append(
                    message
                )

        if use_system_prompt:
            if context:
                retrieved_section = "\n".join(
                    [
                        (
                            "You are a code assistant "
                            "for the Project project."
                        ),
                        (
                            "Use the retrieved project "
                            "sources when relevant."
                        ),
                        (
                            "The sources are ordered by "
                            "reranker relevance."
                        ),
                        (
                            "Do not invent project APIs, "
                            "files, types or behavior."
                        ),
                        (
                            "If the supplied context is "
                            "insufficient, say so explicitly."
                        ),
                        (
                            "When referring to project code, "
                            "mention the source file and symbol."
                        ),
                        "",
                        "Retrieved Project sources:",
                        "",
                        context,
                    ]
                )
            else:
                retrieved_section = "\n".join(
                    [
                        (
                            "You are a code assistant "
                            "for the Project project."
                        ),
                        (
                            "No relevant Project source-code "
                            "fragments were retrieved."
                        ),
                        (
                            "Do not invent project APIs, "
                            "files, types or behavior."
                        ),
                        (
                            "State explicitly that the indexed "
                            "source code is insufficient."
                        ),
                    ]
                )
            system_parts.append(
                retrieved_section
            )
        elif context:
            system_parts.append(
                "\n".join(
                    [
                        "Retrieved Project sources:",
                        "",
                        context,
                    ]
                )
            )

        if not system_parts:
            return other_messages

        return [
            {
                "role": "system",
                "content": "\n\n".join(
                    system_parts
                ),
            },
            *other_messages,
        ]

    async def prepare_messages(
        self,
        messages: list[dict[str, Any]],
        query: str,
        top_k: int | None = None,
        source_type: str | None = None,
        language: str | None = None,
        path_prefix: str | None = None,
        use_system_prompt: bool = True,

    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        results = await self.retrieve(
            query=query,
            source_type=source_type,
            language=language,
            path_prefix=path_prefix,

            top_k=top_k,
        )

        context, sources = (
            self.build_context(results)
        )

        enriched = self.enrich_messages(
            original_messages=messages,
            context=context,
            use_system_prompt=use_system_prompt,
        )

        return enriched, sources

    async def ask(
        self,
        question: str,
        top_k: int | None = None,
        source_type: str | None = None,
        language: str | None = None,
        path_prefix: str | None = None,

    ) -> dict[str, Any]:
        results = await self.retrieve(
            query=question,
            source_type=source_type,
            language=language,
            path_prefix=path_prefix,

            top_k=top_k,
        )

        context, sources = (
            self.build_context(results)
        )

        messages = self.enrich_messages(
            original_messages=[
                {
                    "role": "user",
                    "content": question,
                }
            ],
            context=context,
        )

        timeout = httpx.Timeout(
            REQUEST_TIMEOUT
        )

        async with httpx.AsyncClient(
            timeout=timeout,
        ) as client:
            with track_dependency("llm"):
                response = await client.post(
                    LLM_URL,
                    json=apply_llm_defaults(
                        {
                            "model": LLM_MODEL,
                            "messages": messages,
                            "stream": False,
                        }
                    ),
                )

                response.raise_for_status()

        payload = response.json()
        choices = payload.get("choices")

        if (
            not isinstance(choices, list)
            or not choices
        ):
            raise RuntimeError(
                "LLM response has no choices"
            )

        answer = (
            choices[0]
            .get("message", {})
            .get("content", "")
        )

        return {
            "answer": answer,
            "sources": sources,
        }


rag_service = RagService()


#!/usr/bin/env python3

import json
import os
from pathlib import Path
from typing import Any

import faiss
import httpx
import numpy as np


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
    True,
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


class RagService:
    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []

    def load(self) -> None:
        if not INDEX_FILE.exists():
            raise RuntimeError(
                f"FAISS index not found: {INDEX_FILE}"
            )

        if not METADATA_FILE.exists():
            raise RuntimeError(
                "Metadata file not found: "
                f"{METADATA_FILE}"
            )

        self.index = faiss.read_index(
            str(INDEX_FILE)
        )

        self.metadata = self._load_metadata(
            METADATA_FILE
        )

        if self.index.ntotal != len(self.metadata):
            raise RuntimeError(
                "Index/metadata mismatch: "
                f"{self.index.ntotal} "
                f"!= {len(self.metadata)}"
            )

        print(
            "Loaded Project RAG index: "
            f"documents={self.index.ntotal}, "
            f"dimensions={self.index.d}, "
            f"rerank_enabled={RERANK_ENABLED}, "
            f"rerank_candidates={RERANK_CANDIDATES}"
        )

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
    def document_count(self) -> int:
        if self.index is None:
            return 0

        return int(self.index.ntotal)

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
    def _faiss_fallback(
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for candidate in candidates[:top_k]:
            item = dict(candidate)
            item["score"] = item["faiss_score"]
            results.append(item)

        return results

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        query = query.strip()

        if not query:
            raise ValueError(
                "Query must not be empty"
            )

        if self.index is None:
            raise RuntimeError(
                "FAISS index is not loaded"
            )

        limit = self._normalize_top_k(
            top_k
        )

        candidate_count = (
            self._candidate_count(limit)
            if RERANK_ENABLED
            else limit
        )

        timeout = httpx.Timeout(
            REQUEST_TIMEOUT
        )

        async with httpx.AsyncClient(
            timeout=timeout,
        ) as client:
            vector = await self.get_embedding(
                client,
                query,
            )

            scores, indices = self.index.search(
                vector,
                candidate_count,
            )

            candidates: list[
                dict[str, Any]
            ] = []

            for score, item_index in zip(
                scores[0],
                indices[0],
            ):
                item_index = int(item_index)
                faiss_score = float(score)

                if item_index < 0:
                    continue

                if item_index >= len(
                    self.metadata
                ):
                    continue

                if faiss_score < MIN_FAISS_SCORE:
                    continue

                metadata_item = self.metadata[
                    item_index
                ]

                candidates.append(
                    self._metadata_to_result(
                        item=metadata_item,
                        item_index=item_index,
                        faiss_score=faiss_score,
                    )
                )

            if not candidates:
                return []

            if not RERANK_ENABLED:
                return self._faiss_fallback(
                    candidates,
                    limit,
                )

            try:
                reranked = await self.rerank(
                    client=client,
                    query=query,
                    candidates=candidates,
                    top_k=limit,
                )

                if reranked:
                    return reranked

                if not RERANK_FALLBACK:
                    return []

                print(
                    "Reranker returned no results; "
                    "falling back to FAISS ranking"
                )

            except (
                httpx.HTTPError,
                RuntimeError,
                ValueError,
            ) as error:
                if not RERANK_FALLBACK:
                    raise

                print(
                    "Rerank failed; "
                    "falling back to FAISS: "
                    f"{error}"
                )

            return self._faiss_fallback(
                candidates,
                limit,
            )

    @staticmethod
    def build_context(
        results: list[dict[str, Any]],
    ) -> tuple[
        str,
        list[dict[str, Any]],
    ]:
        chunks: list[str] = []
        sources: list[dict[str, Any]] = []
        total_chars = 0

        for position, result in enumerate(
            results,
            start=1,
        ):
            rerank_score = result.get(
                "rerank_score"
            )

            faiss_score = result.get(
                "faiss_score"
            )

            score_lines: list[str] = []

            if rerank_score is not None:
                score_lines.append(
                    "Rerank relevance: "
                    f"{float(rerank_score):.6f}"
                )

            if faiss_score is not None:
                score_lines.append(
                    "FAISS similarity: "
                    f"{float(faiss_score):.6f}"
                )

            chunk = "\n".join(
                    (
                        "Source type: "
                        f"{result.get('source_type', 'code')}"
                    ),

                [
                    f"[Source {position}]",
                    *score_lines,
                    (
                        "Language: "
                        f"{result['language']}"
                    ),
                    (
                        "File: "
                        f"{result['file']}"
                    ),
                    (
                        "Symbol: "
                        f"{result['symbol']}"
                    ),
                    (
                        "Signature: "
                        f"{result['signature']}"
                    ),
                    (
                        "Lines: "
                        f"{result['start_line']}-"
                        f"{result['end_line']}"
                    ),
                    "Code:",
                    str(result["code"]),
                ]
            )

            remaining = (
                MAX_CONTEXT_CHARS
                - total_chars
            )

            if remaining <= 0:
                break

            if len(chunk) > remaining:
                chunk = chunk[:remaining]

            chunks.append(chunk)
            total_chars += len(chunk)

            sources.append(
                {
                    "score": result["score"],
                    "rerank_score": (
                        result.get(
                            "rerank_score"

                        )
                    ),
                    "faiss_score": (
                        result.get(
                            "faiss_score"
                        )
                    ),
                    "file": result["file"],
                    "source_type": result.get("source_type", "code"),

                    "symbol": result["symbol"],
                    "start_line": (
                        result["start_line"]
                    ),
                    "end_line": (
                        result["end_line"]
                    ),
                }
            )

        return (
            "\n\n---\n\n".join(chunks),
            sources,
        )

    @staticmethod
    def enrich_messages(
        original_messages: list[
            dict[str, Any]
        ],
        context: str,
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
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        results = await self.retrieve(
            query=query,
            top_k=top_k,
        )

        context, sources = (
            self.build_context(results)
        )

        enriched = self.enrich_messages(
            original_messages=messages,
            context=context,
        )

        return enriched, sources

    async def ask(
        self,
        question: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        results = await self.retrieve(
            query=question,
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
            response = await client.post(
                LLM_URL,
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "stream": False,
                    "temperature": 0.2,
                },
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


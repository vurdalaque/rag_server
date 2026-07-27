#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import requests


INDEX_FILE = Path("rag_index.faiss")
METADATA_FILE = Path("rag_metadata.jsonl")

EMBEDDING_URL = "http://192.168.10.250:8002/v1/embeddings"
EMBEDDING_MODEL = "qwen3-embedding"

REQUEST_TIMEOUT = 120


def load_metadata() -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []

    with METADATA_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                metadata.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON in metadata at line {line_number}: {error}"
                ) from error

    return metadata


def get_embedding(text: str) -> np.ndarray:
    response = requests.post(
        EMBEDDING_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": [text],
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    result = response.json()
    data = result.get("data")

    if not isinstance(data, list) or len(data) != 1:
        raise RuntimeError("Unexpected embeddings response")

    vector = np.asarray(
        data[0]["embedding"],
        dtype=np.float32,
    ).reshape(1, -1)

    faiss.normalize_L2(vector)

    return vector


def shorten_code(code: str, max_chars: int) -> str:
    if max_chars <= 0 or len(code) <= max_chars:
        return code

    return code[:max_chars].rstrip() + "\n... [truncated]"


def search(
    index: faiss.Index,
    metadata: list[dict[str, Any]],
    query: str,
    top_k: int,
) -> list[tuple[float, dict[str, Any]]]:
    query_vector = get_embedding(query)

    if query_vector.shape[1] != index.d:
        raise RuntimeError(
            f"Embedding dimensions mismatch: "
            f"query={query_vector.shape[1]}, index={index.d}"
        )

    scores, indices = index.search(query_vector, top_k)

    results: list[tuple[float, dict[str, Any]]] = []

    for score, item_index in zip(scores[0], indices[0]):
        if item_index < 0:
            continue

        if item_index >= len(metadata):
            raise RuntimeError(
                f"FAISS index points outside metadata: {item_index}"
            )

        results.append(
            (
                float(score),
                metadata[item_index],
            )
        )

    return results


def print_results(
    results: list[tuple[float, dict[str, Any]]],
    max_code_chars: int,
) -> None:
    if not results:
        print("No results")
        return

    for position, (score, item) in enumerate(results, start=1):
        file_path = item.get("file", "")
        language = item.get("language", "")
        name = item.get("name", "")
        full_name = item.get("full_name", "")
        detail = item.get("detail", "")
        start_line = item.get("start_line")
        end_line = item.get("end_line")
        code = item.get("code", "")

        print("=" * 100)
        print(f"Result:    {position}")
        print(f"Score:     {score:.6f}")
        print(f"Language:  {language}")
        print(f"File:      {file_path}")
        print(f"Symbol:    {name}")

        if full_name and full_name != name:
            print(f"Full name: {full_name}")

        if detail:
            print(f"Signature: {detail}")

        if start_line is not None or end_line is not None:
            print(f"Lines:     {start_line}-{end_line}")

        print("-" * 100)
        print(shorten_code(str(code), max_code_chars))

    print("=" * 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search the Project FAISS index"
    )

    parser.add_argument(
        "query",
        nargs="+",
        help="Search query",
    )

    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of results, default: 5",
    )

    parser.add_argument(
        "--max-code-chars",
        type=int,
        default=3000,
        help="Maximum code characters per result; 0 means unlimited",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero")

    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Index file not found: {INDEX_FILE}")

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    query = " ".join(args.query).strip()

    if not query:
        raise ValueError("Query must not be empty")

    index = faiss.read_index(str(INDEX_FILE))
    metadata = load_metadata()

    if index.ntotal != len(metadata):
        raise RuntimeError(
            f"Index/metadata mismatch: "
            f"index={index.ntotal}, metadata={len(metadata)}"
        )

    print(f"Query: {query}")
    print(f"Index documents: {index.ntotal}")
    print()

    results = search(
        index=index,
        metadata=metadata,
        query=query,
        top_k=args.top_k,
    )

    print_results(
        results=results,
        max_code_chars=args.max_code_chars,
    )


if __name__ == "__main__":
    main()


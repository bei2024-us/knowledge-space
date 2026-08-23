from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import numpy as np
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
LOCAL_TEMP_DIR = APP_DIR / "data" / "tmp"
LOCAL_TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TMP", str(LOCAL_TEMP_DIR))
os.environ.setdefault("TEMP", str(LOCAL_TEMP_DIR))
os.environ.setdefault("TMPDIR", str(LOCAL_TEMP_DIR))

try:
    from FlagEmbedding import BGEM3FlagModel
except Exception:  # pragma: no cover - optional dependency
    BGEM3FlagModel = None


EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3").strip()
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
EMBEDDING_MAX_LENGTH = int(os.getenv("EMBEDDING_MAX_LENGTH", "8192"))
EMBEDDING_USE_FP16 = os.getenv("EMBEDDING_USE_FP16", "auto").strip().lower()
EMBEDDING_DEVICES = [part.strip() for part in os.getenv("EMBEDDING_DEVICES", "").split(",") if part.strip()]


@lru_cache(maxsize=1)
def load_embedding_model():
    if BGEM3FlagModel is None:
        return None
    kwargs: dict[str, Any] = {"use_fp16": _resolve_fp16()}
    if EMBEDDING_DEVICES:
        kwargs["devices"] = EMBEDDING_DEVICES
    try:
        return BGEM3FlagModel(EMBEDDING_MODEL_NAME, **kwargs)
    except Exception:
        return None


def _resolve_fp16() -> bool:
    if EMBEDDING_USE_FP16 in {"1", "true", "yes", "on"}:
        return True
    if EMBEDDING_USE_FP16 in {"0", "false", "no", "off"}:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def embedding_backend_name() -> str | None:
    if load_embedding_model() is None:
        return None
    return f"FlagEmbedding:{EMBEDDING_MODEL_NAME}"


def encode_dense_texts(texts: list[str]) -> np.ndarray | None:
    model = load_embedding_model()
    if model is None or not texts:
        return None

    try:
        output = model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            max_length=EMBEDDING_MAX_LENGTH,
        )
        dense_vecs = output["dense_vecs"] if isinstance(output, dict) else output
        matrix = np.asarray(dense_vecs, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return _normalize_rows(matrix)
    except Exception:
        return None


def embed_chunks(chunk_texts: list[str]) -> np.ndarray | None:
    return encode_dense_texts(chunk_texts)


def embed_query(query: str) -> np.ndarray | None:
    matrix = encode_dense_texts([query])
    if matrix is None or matrix.size == 0:
        return None
    return matrix[0]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def store_chunk_embeddings(conn, chunk_ids: list[int], chunk_texts: list[str]) -> bool:
    vectors = embed_chunks(chunk_texts)
    if vectors is None:
        return False

    payloads = []
    for chunk_id, vector in zip(chunk_ids, vectors):
        payloads.append((chunk_id, EMBEDDING_MODEL_NAME, json.dumps(vector.tolist(), ensure_ascii=False)))

    conn.executemany(
        """
        INSERT OR REPLACE INTO chunk_embeddings(chunk_id, model, embedding)
        VALUES(?, ?, ?)
        """,
        payloads,
    )
    return True


def backfill_space_embeddings(conn, space_id: int, batch_size: int = 32) -> int:
    if load_embedding_model() is None:
        return 0

    rows = conn.execute(
        """
        SELECT c.id AS chunk_id,
               c.text
        FROM chunks c
        LEFT JOIN chunk_embeddings e
          ON e.chunk_id = c.id
         AND e.model = ?
        WHERE c.space_id = ?
          AND e.chunk_id IS NULL
        ORDER BY c.id
        LIMIT ?
        """,
        (EMBEDDING_MODEL_NAME, space_id, batch_size),
    ).fetchall()
    if not rows:
        return 0

    chunk_ids = [int(row["chunk_id"]) for row in rows]
    chunk_texts = [row["text"] for row in rows]
    store_chunk_embeddings(conn, chunk_ids, chunk_texts)
    return len(rows)


def load_space_embeddings(conn, space_id: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT c.id AS chunk_id,
               c.document_id,
               c.location_label,
               c.text,
               d.filename,
               d.folder_id,
               f.name AS folder_name,
               d.file_type,
               e.embedding
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        LEFT JOIN folders f ON f.id = d.folder_id
        JOIN chunk_embeddings e ON e.chunk_id = c.id
        WHERE c.space_id = ?
          AND e.model = ?
        ORDER BY c.id
        """,
        (space_id, EMBEDDING_MODEL_NAME),
    ).fetchall()
    results: list[dict] = []
    for row in rows:
        item = dict(row)
        item["vector"] = np.asarray(json.loads(item.pop("embedding")), dtype=np.float32)
        results.append(item)
    return results


def semantic_rank_chunks(query: str, rows: list[dict], limit: int) -> list[dict]:
    query_vector = embed_query(query)
    if query_vector is None or not rows:
        return []

    vectors = []
    usable_rows = []
    for row in rows:
        vector = row.get("vector")
        if vector is None:
            continue
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1 or vector.size != query_vector.size:
            continue
        vectors.append(vector)
        usable_rows.append(row)

    if not vectors:
        return []

    matrix = _normalize_rows(np.vstack(vectors))
    scores = matrix @ query_vector

    ranked: list[dict] = []
    for row, score in zip(usable_rows, scores):
        item = {key: value for key, value in row.items() if key != "vector"}
        item["semantic_score"] = float(score)
        ranked.append(item)

    ranked.sort(key=lambda item: item["semantic_score"], reverse=True)
    return ranked[:limit]

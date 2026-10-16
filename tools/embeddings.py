import json
from pathlib import Path
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from sentence_transformers import SentenceTransformer

from tools.chunker import Chunk, chunk_directory
from tools.config import CORPUS_DIRS, EMBEDDING_MODEL, INDEX_DIR, TOP_K

_model = None  # 缓存已加载的模型，避免重复加载


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,   # 关键：归一化 → 点积即余弦
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")

def build_index(corpus_dirs: list[str] | None = None) -> dict:
    corpus_dirs = corpus_dirs or CORPUS_DIRS
    chunks: list[Chunk] = []
    for directory in corpus_dirs:
        chunks.extend(chunk_directory(directory))   # 复用你写好的 chunker

    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vectors)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps([c.model_dump() for c in chunks], ensure_ascii=False),
        encoding="utf-8",
    )
    return {"chunks": len(chunks), "dim": vectors.shape[1]}

def load_index() -> tuple[list[dict], np.ndarray]:
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(INDEX_DIR / "vectors.npy")
    return chunks, vectors


def search(query: str, top_k: int = TOP_K) -> list[dict]:
    chunks, vectors = load_index()
    q = embed_texts([query])[0]          # 查询也归一化
    scores = vectors @ q                 # 归一化后点积 = 余弦
    order = np.argsort(scores)[::-1][:top_k]
    return [
        {**chunks[i], "score": float(scores[i])}
        for i in order
    ]
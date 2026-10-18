from pydantic import BaseModel, Field

from tools.config import INDEX_DIR, RRF_CANDIDATES, RRF_K, TOP_K
from tools import dense_search, sparse_search


class HybridSearchInput(BaseModel):
    """混合检索工具的结构化输入。"""

    query: str = Field(
        description="要检索的问题或关键词。复杂问题建议拆成多个子问题分别检索。"
    )
    top_k: int = Field(
        default=TOP_K,
        ge=1,
        le=20,
        description="融合后返回最相关的块数，默认 5。",
    )


def _rrf_fuse(dense: list[dict], sparse: list[dict], top_k: int) -> list[dict]:
    """把两路结果按 RRF 融合。块用 chunks.json 里的下标 id 对齐。"""
    fused: dict[int, dict] = {}
    for rank, item in enumerate(dense, start=1):
        entry = fused.setdefault(item["id"], dict(item))
        entry["dense_score"] = item["score"]
        entry["rrf"] = entry.get("rrf", 0.0) + 1.0 / (RRF_K + rank)
    for rank, item in enumerate(sparse, start=1):
        entry = fused.setdefault(item["id"], dict(item))
        entry["sparse_score"] = item["score"]
        entry["rrf"] = entry.get("rrf", 0.0) + 1.0 / (RRF_K + rank)
    for entry in fused.values():
        entry.pop("score", None)   # 去掉单路原始 score，避免和 dense_score/sparse_score 混淆
    ordered = sorted(fused.values(), key=lambda e: -e["rrf"])
    return ordered[:top_k]


def hybrid_search_tool(request: HybridSearchInput) -> list[dict]:
    """在已构建的索引上做混合检索，返回融合后的 top-k 块及其来源。"""
    if not (INDEX_DIR / "chunks.json").exists() or not (INDEX_DIR / "vectors.npy").exists():
        raise FileNotFoundError("知识库索引尚未构建，请先运行 python index.py")
    if not (INDEX_DIR / "sparse.json").exists():
        raise FileNotFoundError("词法索引尚未构建，请先运行 python index.py")
    dense = dense_search.search(request.query, top_k=RRF_CANDIDATES)
    sparse = sparse_search.search(request.query, top_k=RRF_CANDIDATES)
    return _rrf_fuse(dense, sparse, request.top_k)
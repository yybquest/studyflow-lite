"""语义检索工具：把查询转成知识库中最相关的 top-k 个块（不生成答案）。"""
from pydantic import BaseModel, Field

from tools.config import INDEX_DIR, TOP_K
from tools.embeddings import search


class SearchKnowledgeInput(BaseModel):
    """语义检索工具的结构化输入。"""

    query: str = Field(
        description="要检索的问题或关键词。复杂问题建议拆成多个子问题分别检索。"
    )
    top_k: int = Field(
        default=TOP_K,
        ge=1,
        le=20,
        description="返回最相关的块数，默认 5。",
    )


def search_knowledge_tool(request: SearchKnowledgeInput) -> list[dict]:
    """在已构建的索引中做语义检索，返回 top-k 个最相关块及其来源。"""
    if not (INDEX_DIR / "chunks.json").exists() or not (INDEX_DIR / "vectors.npy").exists():
        raise FileNotFoundError("知识库索引尚未构建，请先运行 python index.py")
    return search(request.query, top_k=request.top_k)

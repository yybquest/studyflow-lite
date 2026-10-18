import json
import math
import re
from collections import Counter

from tools.config import INDEX_DIR, TOP_K

_ASCII_WORD = re.compile(r"[a-z0-9]+")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """把文本切成词法单元：英文/数字按单词，中文按二字组，单字兜底。

    单字保证单个汉字也能命中；二字组（bigram）提高精确度、减少单字歧义。
    """
    text = text.casefold()
    tokens = list(_ASCII_WORD.findall(text))
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)                              # 单字查询兜底
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))          # 二字组
    return tokens


def _vectorize(text: str, idf: dict[str, float]) -> dict[str, float]:
    """把文本转成 L2 归一化的 TF-IDF 稀疏向量（只保留出现在 idf 里的词）。"""
    tf = Counter(tokenize(text))
    vec: dict[str, float] = {}
    for term, count in tf.items():
        if term in idf:
            vec[term] = (1.0 + math.log(count)) * idf[term]
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm:
        vec = {t: v / norm for t, v in vec.items()}
    return vec


class SparseIndex:
    """对 chunks.json 里的每个块建 TF-IDF 向量，供词法检索。"""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.idf: dict[str, float] = {}
        self.doc_vectors: list[dict[str, float]] = []

    def build(self) -> None:
        n = len(self.chunks)
        df: Counter = Counter()
        for chunk in self.chunks:
            df.update(set(tokenize(chunk["text"])))   # 文档频率：一个词在一篇里出现多次也只计 1

        max_df = max(2, 0.10 * n)   # 出现在 >10% 文档里的近停用词（什么/作用…）直接丢弃，但至少保留出现在 2 篇以内的词
        self.idf = {
            t: math.log((1 + n) / (1 + c)) + 1.0
            for t, c in df.items()
            if c <= max_df
        }
        self.doc_vectors = [_vectorize(chunk["text"], self.idf) for chunk in self.chunks]

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        q = _vectorize(query, self.idf)
        if not q:          # 查询词全部不在词表里 → 直接返回空，避免返回一堆 0 分
            return []
        scores = [
            sum(weight * q.get(term, 0.0) for term, weight in doc.items())
            for doc in self.doc_vectors
        ]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [dict(self.chunks[i], id=i, score=scores[i]) for i in order]


def build_sparse_index() -> dict:
    """从 chunks.json 读取块，构建并落盘稀疏索引（构建期调用）。"""
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    index = SparseIndex(chunks)
    index.build()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (INDEX_DIR / "sparse.json").write_text(
        json.dumps({"idf": index.idf, "docs": index.doc_vectors}, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"terms": len(index.idf), "docs": len(index.doc_vectors)}


def load_sparse_index(chunks: list[dict]) -> SparseIndex:
    """读取已落盘的稀疏索引（查询期调用）。"""
    index = SparseIndex(chunks)
    data = json.loads((INDEX_DIR / "sparse.json").read_text(encoding="utf-8"))
    index.idf = data["idf"]
    index.doc_vectors = data["docs"]
    return index


def search(query: str, top_k: int = TOP_K) -> list[dict]:
    """词法检索入口：加载 chunks + 稀疏索引后返回 top-k 块。"""
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    index = load_sparse_index(chunks)
    return index.search(query, top_k=top_k)
# 两段式术语提取（超大论文用）— 设计草稿，暂未接入

> 适用场景：单篇论文的 markdown 过大（接近或超出模型上下文），一次性 `_extract_terms(client, content)` 吃不下、或明显漏掉中间章节的术语时，用这个替换。

## 设计思路

- **第一遍（map，分节提取）**：把论文按「节」拆开，逐节从**原文**提取术语——不分类、不限量，追求召回全；
- **合并去重**：把所有节的术语合并，按 `_term_key` 去重；
- **第二遍（reduce，全局分类）**：把去重后的唯一术语列表 + 论文标题，一次性分类成「通用术语 / 本文专属名词」，并只保留最重要的通用术语。

核心思想一句话：**提取分节（可扩展、召回全），分类全局（判断准）**。

## 代码（粘进 summarizer.py 即可，复用其中的 _chat / _group_by_heading / _split_long_text / _term_name / _term_key）

```python
def _extract_raw_terms(client, text: str) -> str:
    """第一遍的单节提取：只提术语、不分类、不限量。"""
    prompt = (
        "从下面的文本中提取所有专业术语/名词，为每个写一句简洁中文解释。"
        "术语名用原文英文。只输出术语列表，每行一个，格式：- **术语**：解释\n\n"
        f"文本：\n{text}"
    )
    return _chat(client, [{"role": "user", "content": prompt}])


def _extract_raw_terms_from_sections(client, chunks: list[Chunk]) -> list[str]:
    """第一遍（map）：按节从原文提取术语，返回原始术语行的列表。"""
    raw_lines: list[str] = []
    for heading, text in _group_by_heading(chunks):
        for piece in _split_long_text(text, SECTION_MAX_CHARS):
            for line in _extract_raw_terms(client, piece).splitlines():
                if _term_name(line):
                    raw_lines.append(line.strip())
    return raw_lines


def _dedup_raw_terms(raw_lines: list[str]) -> list[str]:
    """合并 + 去重：按 _term_key 去重，保留每个术语第一次出现的行。"""
    seen: set[str] = set()
    kept: list[str] = []
    for line in raw_lines:
        name = _term_name(line)
        key = _term_key(name) if name else None
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        kept.append(line)
    return kept


def _classify_terms(client, term_lines: list[str], title: str) -> str:
    """第二遍（reduce）：把合并后的唯一术语一次性分类为 通用/专属。"""
    term_list = "\n".join(term_lines)
    prompt = (
        "下面是某篇论文里提取出的所有术语。请把它们分成两组：\n\n"
        "【通用术语】：跨论文、跨领域都成立的通用技术概念（如 RAG、MoE、F1）。"
        "只保留最重要的 15~20 个，同一概念只给一个名称，术语名用原文英文。\n"
        "【本文专属名词】：只有这篇论文才有的系统名、框架名、模块名、基准名。\n\n"
        "每组内每行一个，格式：- **术语**：一句简洁中文解释。\n"
        "只输出这两组，不要其他内容。\n\n"
        f"论文标题：{title}\n术语列表：\n{term_list}"
    )
    return _chat(client, [{"role": "user", "content": prompt}])


def extract_terms_two_pass(client, chunks: list[Chunk], title: str) -> str:
    """两段式术语提取总入口。返回与 _extract_terms 相同格式的两组术语。"""
    raw_lines = _extract_raw_terms_from_sections(client, chunks)
    unique_lines = _dedup_raw_terms(raw_lines)
    return _classify_terms(client, unique_lines, title)
```

## 接入方法（将来启用时）

在 `summarize_paper` 里，把：

```python
terms_md = _extract_terms(client, content)
```

换成：

```python
title = chunks[0].heading.split(" > ")[0] if chunks else Path(md_path).stem
terms_md = extract_terms_two_pass(client, chunks, title)
```

之后的 `_split_terms`、`_dedup_terms`、写盘逻辑**都不变**（因为 `extract_terms_two_pass` 返回的两组格式和 `_extract_terms` 完全一样）。

## 注意点

- **调用次数多**：一篇论文约 20~30 节 → 20~30 次「第一遍」调用 + 1 次「第二遍」分类，总延迟和 token 成本高于单次提取；
- **只在超大论文时用**：中小论文仍用现有单次 `_extract_terms` 更省；
- **去重发生在合并后**，不是每节内；
- **第一遍不限量**：追求召回全，靠第二遍「top 15~20」收敛数量。

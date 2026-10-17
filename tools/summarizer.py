import os
import re
from tools.chunker import Chunk,chunk_text
from pathlib import Path
from tools.file_reader import ReadFileInput, read_file_content
#from tools.embeddings import embed_texts 
from tools.config import PROJECT_ROOT, GLOSSARY_PATH

SECTION_MAX_CHARS = 8000
SHORT_DOC_CHARS = 20000
# 术语行格式：- **术语**：解释
_TERM_RE = re.compile(r"^\s*-\s*\*\*(.+?)\*\*")

def _chat(client, messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat"),
        messages=messages,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return response.choices[0].message.content or ""

def _group_by_heading(chunks: list[Chunk]) -> list[tuple[str, str]]:
    """把连续同标题的块合并成「节」，返回 [(标题, 合并文本), ...]。"""
    groups: list[list[str]] = []
    for c in chunks:
        if groups and groups[-1][0] == c.heading:
            groups[-1][1] += "\n\n" + c.text
        else:
            groups.append([c.heading, c.text])
    return groups

def _split_long_text(text: str, max_chars: int) -> list[str]:
    """把超长文本按 max_chars 切成多段，尽量在换行处切，避免切碎句子。"""
    if len(text) <= max_chars:
        return [text]
    parts = []
    while len(text) > max_chars:
        cut = text.rfind("\n", 0, max_chars)   # 在 max_chars 范围内找最后一个换行
        cut = cut if cut > 0 else max_chars     # 找不到换行就硬切
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    if text:
        parts.append(text)
    return parts

def _summarize_chunk(client, heading: str, text: str) -> str:
    prompt = (
        "你是论文阅读助手。下面是文档的一节。请用 2-3 句话概括这一节的核心内容，"
        "突出关键方法、结论或技术点，不要复述原文。\n\n"
        f"标题：{heading or '正文'}\n正文：\n{text}"
    )
    return _chat(client, [{"role": "user", "content": prompt}])

def _summarize_sections(client, chunks: list[Chunk]) -> str:
    parts = []
    for heading, text in _group_by_heading(chunks):
        # 超长节再切成多段分别总结（在换行处切，不切碎句子）
        for piece in _split_long_text(text, SECTION_MAX_CHARS):
            parts.append(f"【{heading or '正文'}】\n" + _summarize_chunk(client, heading, piece))
    return "\n\n".join(parts)

def _summarize_overall(client, source_text: str) -> str:
    prompt = (
        "你是论文阅读助手。下面是文档的内容（或分节总结）。请据此生成一份结构化的阅读笔记（Markdown），"
        "必须包含以下五个小节：\n"
        "## 一句话总结\n## 研究问题\n## 方法\n## 贡献\n## 关键技术\n\n"
        "要求：只依据给出的内容，不要编造；关键技术要具体、可操作。\n\n"
        f"内容：\n{source_text}"
    )
    return _chat(client, [{"role": "user", "content": prompt}])

def _term_key(name: str) -> str:
    """把术语名归一化成可比对的 key：小写、去括号注解、去所有分隔符。"""
    name = name.strip().lower()
    name = re.sub(r"[（(][^）)]*[）)]", "", name)   # 去中英文括号注解（如 (PA)）
    name = re.sub(r"[-_/\\\s]+", "", name)          # 去掉连字符/下划线/斜杠/空白
    return name

def _term_name(line: str) -> str | None:
    """从一行 '- **术语**：解释' 里提取术语名（原样）。"""
    m = _TERM_RE.match(line.strip())
    return m.group(1).strip() if m else None

def _split_terms(terms_md: str) -> tuple[str, str]:
    """把术语输出拆成 (通用术语, 本文专属名词) 两部分。"""
    general_lines: list[str] = []
    specific_lines: list[str] = []
    current = "general"          # 没找到标记时，默认按通用处理
    for line in terms_md.splitlines():
        if "通用术语" in line:
            current = "general"
            continue
        elif "专属" in line:
            current = "specific"
            continue
        if line.strip():
            (general_lines if current == "general" else specific_lines).append(line.strip())
    return "\n".join(general_lines), "\n".join(specific_lines)

def _existing_term_keys() -> set[str]:
    """读出现有 glossary 里所有术语的归一化 key。"""
    if not GLOSSARY_PATH.exists():
        return set()
    keys = set()
    for line in GLOSSARY_PATH.read_text(encoding="utf-8").splitlines():
        name = _term_name(line)
        if name:
            keys.add(_term_key(name))
    return keys

def _dedup_terms(terms_md: str, existing_keys: set[str]) -> str:
    """按归一化 key 去重，返回只含新术语的多行文本。"""
    kept = []
    for line in terms_md.splitlines():
        name = _term_name(line)
        if name and _term_key(name) in existing_keys:
            continue
        if line.strip():
            kept.append(line.strip())
    return "\n".join(kept)

def _extract_terms(client, text: str) -> str:
    prompt = (
        "从下面的文本中提取最重要的专业术语，分成两组：\n\n"
        "【通用术语】：跨论文、跨领域都成立的通用技术概念（如 RAG、MoE、F1、混合检索），术语名用原文英文。\n"
        "【本文专属名词】：只有这篇论文才有的系统名、框架名、模块名、基准名（如 KnowForge、Locator、X-AgentRed）。\n\n"
        "每组内每行一个，格式：- **术语**：一句简洁中文解释。\n"
        "只输出这两组，不要其他内容。\n\n"
        f"文本：\n{text}"
    )
    return _chat(client, [{"role": "user", "content": prompt}])

""" 以下是语义去重，可用但效率较低

def _parse_terms(text: str) -> list[tuple[str, str]]:
    terms = []
    for line in text.splitlines():
        m = _TERM_RE.match(line.strip())
        if m:
            name = m.group(1).strip()
            expl = line.strip().split("**", 2)[-1].lstrip("：: ").strip()
            terms.append((name, expl))
    return terms

def _semantic_dedup_terms(terms_md: str, existing_text: str, threshold: float = 0.75) -> str:
    existing = _parse_terms(existing_text)
    new_terms = _parse_terms(terms_md)
    if not existing:
        return terms_md                      # 空 glossary，全部保留

    existing_vecs = embed_texts([f"{n}：{e}" for n, e in existing])  # 已有术语向量化一次
    kept = []
    for name, expl in new_terms:
        q = embed_texts([f"{name}：{expl}"])[0]
        if (existing_vecs @ q).max() >= threshold:   # 点积=余弦（已归一化）
            continue                                    # 语义重复，跳过
        kept.append(f"- **{name}**：{expl}")
    return "\n".join(kept)

"""

def summarize_paper(client, md_path: str) -> dict:
    content = read_file_content(ReadFileInput(file_path=md_path))   # 复用安全读取
    chunks = chunk_text(content, md_path)                          # 复用切块

    # 自适应：短文档直接单次总结，长文档走 map-reduce
    if len(content) < SHORT_DOC_CHARS:      # SHORT_DOC_CHARS = 20000
        source_text = content               # 全文直接当源
    else:
        source_text = _summarize_sections(client, chunks)   # 先分节总结

    summary_md = _summarize_overall(client, source_text)
    terms_md = _extract_terms(client, content)
    general_terms, specific_terms = _split_terms(terms_md)

    # 组装 summary 文件的术语表：通用 + 专属都写
    terms_section = general_terms
    if specific_terms:
        terms_section += "\n\n### 本文专属名词\n\n" + specific_terms

    # 写 summary 文件：同目录 <原名>-summary.md
    p = Path(md_path)
    out_path = PROJECT_ROOT / p.parent / f"{p.stem}-summary.md"
    out_path.write_text(
        summary_md + "\n\n## 术语表\n\n" + terms_section,
        encoding="utf-8",
    )

    # glossary：只 append 通用术语（去重后）
    existing_keys = _existing_term_keys()
    new_terms = _dedup_terms(general_terms, existing_keys)
    if new_terms:
        with GLOSSARY_PATH.open("a", encoding="utf-8") as f:
            f.write("\n\n" + new_terms)

    return {
        "summary": out_path.relative_to(PROJECT_ROOT).as_posix(),
        "glossary": GLOSSARY_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "new_terms": len(new_terms.splitlines()) if new_terms else 0,
        "skipped_terms": len(general_terms.splitlines()) - (len(new_terms.splitlines()) if new_terms else 0),
    }
from __future__ import annotations

import re
from dataclasses import dataclass

from tools.config import (
    FALLBACK_SECTION_CHARS,
    HEADING_RE,
    MIN_CHUNK_CHARS,
    NUMBER_PREFIX_RE,
    SECTION_SKIP_TITLES,
)

_NUMBERED_RE = re.compile(r"^\d+([\.\s、\)]|$)")

# 这些标题视为「正文起点」；位于它们之前的节视为作者块/前置信息，丢弃
_FRONT_MATTER_TITLES = {"abstract", "introduction", "摘要", "引言"}


@dataclass
class Section:
    """论文的一节：标题、在原文中的行号范围，以及原始文本。"""

    index: int
    title: str
    start_line: int   # 1-based，含
    end_line: int     # 1-based，含
    text: str


def _clean_title(raw: str) -> str:
    """清洗标题：去掉 Markdown 标记、锚点与包裹符号。"""
    return raw.strip().rstrip("#").strip().strip("*`_").strip()


def _is_skip_title(title: str) -> bool:
    """标题是否属于「跳过」的低价值节（参考文献、致谢等）。"""
    last = title.strip().lower()
    last = NUMBER_PREFIX_RE.sub("", last).strip()
    return last in SECTION_SKIP_TITLES


def _is_content_start(title: str) -> bool:
    """标题是否像正文起点（Abstract / Introduction / 编号节）。"""
    stripped = NUMBER_PREFIX_RE.sub("", title.strip()).strip().lower()
    if stripped in _FRONT_MATTER_TITLES:
        return True
    return bool(_NUMBERED_RE.match(title.strip()))


def _drop_front_matter(sections: list[Section]) -> list[Section]:
    """丢弃正文起点之前的节（通常是作者名、单位、邮箱等前置信息）。"""
    for position, section in enumerate(sections):
        if _is_content_start(section.title):
            return sections[position:]
    return sections


def parse_sections(md_text: str, *, min_body_chars: int = MIN_CHUNK_CHARS) -> list[Section]:
    """按 `##` 二级标题把 Markdown 切成节。

    - `###` 及更深的标题归入所属的 `##` 节；
    - 代码围栏内的 `#` 不算标题；
    - 跳过参考文献/致谢等低价值节，以及正文过短的节；
    - 丢弃 Abstract/Introduction 之前的作者块；
    - 解析不出任何可用节时返回 []，由调用方决定是否兜底。
    """
    lines = md_text.splitlines()
    headings: list[tuple[int, str]] = []
    in_fence = False

    for line_index, raw in enumerate(lines):
        if raw.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = HEADING_RE.match(raw)
        if match and len(match.group(1)) == 2:
            headings.append((line_index, _clean_title(match.group(2))))

    sections: list[Section] = []
    for position, (start_index, title) in enumerate(headings):
        end_index = (
            headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        )
        body = "\n".join(lines[start_index + 1:end_index]).strip()

        if _is_skip_title(title):
            continue
        if len(body) < min_body_chars:
            continue

        sections.append(Section(
            index=len(sections),
            title=title or "（无标题）",
            start_line=start_index + 1,
            end_line=end_index,
            text="\n".join(lines[start_index:end_index]).strip(),
        ))

    sections = _drop_front_matter(sections)
    for index, section in enumerate(sections):
        section.index = index
    return sections


def fallback_sections(md_text: str, *, chars: int = FALLBACK_SECTION_CHARS) -> list[Section]:
    """无可靠分节时的兜底：按固定字符数切伪节（行号为近似值）。"""
    text = md_text.strip()
    if not text:
        return []

    sections: list[Section] = []
    for start in range(0, len(text), chars):
        end = min(start + chars, len(text))
        piece = text[start:end].strip()
        if len(piece) < MIN_CHUNK_CHARS:
            continue
        sections.append(Section(
            index=len(sections),
            title=f"第 {len(sections) + 1} 段（无标题兜底）",
            start_line=text.count("\n", 0, start) + 1,
            end_line=text.count("\n", 0, end) + 1,
            text=piece,
        ))
    return sections

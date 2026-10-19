from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

from tools.config import (
    BLOCKED_DIRECTORY_NAMES,
    HEADING_RE,
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    NUMBER_PREFIX_RE,
    OVERLAP_CHARS,
    PROJECT_ROOT,
    SECTION_SKIP_TITLES,
    TEXT_SUFFIXES,
)
from tools.file_reader import ReadFileInput, read_file_content

# 可追溯
class Chunk(BaseModel):
    """一个文本块，携带来源与定位信息，供后续索引与检索使用。"""

    text: str
    source_file: str
    heading: str = ""
    start_line: int
    end_line: int


def _section_len(section: list[tuple[int, str]]) -> int:
    """估算一个节（若干行）的总字符数，换行按 1 字符计入。"""
    return sum(len(text) + 1 for _, text in section)


def _make_chunk(lines: list[tuple[int, str]], heading: str, source_file: str) -> Chunk:
    text = "\n".join(t for _, t in lines).strip()
    return Chunk(
        text=text,
        source_file=source_file,
        heading=heading,
        start_line=lines[0][0],   # 可追溯：节第一行行号
        end_line=lines[-1][0],    # 可追溯：节最后一行行号
    )


def _is_skip_title(heading: str) -> bool:
    """噪音过滤——判断标题（面包屑末段）是否属于要跳过的低价值节。"""
    last = heading.split(" > ")[-1].strip().lower()
    # 去掉 "5." "3.2" 之类的前缀编号后再比较
    last = NUMBER_PREFIX_RE.sub("", last).strip()
    return last in SECTION_SKIP_TITLES


def _split_long(section: list[tuple[int, str]], heading: str, source_file: str) -> list[Chunk]:
    """结构感知（兜底）——把超长节按字符窗口 + 重叠切成多个块（单行过长也能切）。"""
    # 可追溯——拼接整节文本，并记录每个字符所属的行号，用于回溯
    text_parts: list[str] = []
    line_numbers: list[int] = []
    for line_no, t in section:
        text_parts.append(t)
        line_numbers.extend([line_no] * len(t))
        text_parts.append("\n")
        line_numbers.append(line_no)

    full = "".join(text_parts)
    chunks: list[Chunk] = []
    start = 0
    total = len(full)
    while start < total:
        end = min(start + MAX_CHUNK_CHARS, total)
        piece = full[start:end].strip()
        if len(piece) >= MIN_CHUNK_CHARS:   # 噪音过滤：丢弃过短碎片
            chunks.append(Chunk(
                text=piece,
                source_file=source_file,
                heading=heading,
                start_line=line_numbers[start],   # 可追溯：窗口起点字符 → 行号
                end_line=line_numbers[end - 1],   # 可追溯：窗口终点字符 → 行号
            ))
        if end >= total:
            break
        start = end - OVERLAP_CHARS

    return chunks


def _emit_section(section: list[tuple[int, str]], heading: str, source_file: str) -> list[Chunk]:
    """把一个节（标题 + 内容行）转成一个或多个块。"""
    if not section:
        return []
    if _is_skip_title(heading):   # 噪音过滤：跳过参考文献等低价值节
        return []
    if _section_len(section) <= MAX_CHUNK_CHARS:
        chunk = _make_chunk(section, heading, source_file)
        # 噪音过滤：丢弃过短碎片
        return [chunk] if len(chunk.text) >= MIN_CHUNK_CHARS else []
    # 结构感知：只有超长节才走字符窗口兜底（正常节直接一个块）
    return _split_long(section, heading, source_file)


def chunk_text(text: str, source_file: str) -> list[Chunk]:
    """把 Markdown 文本切成块列表——核心切块逻辑。
    设计为纯函数（无副作用、确定性）：不读文件、不写文件，只对入参字符串做处理，
    因此既被 chunk_markdown_file 调用，也能在测试里直接喂字符串验证。
    """
    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []  # 结构感知：(级别, 标题)，维护标题层级
    section: list[tuple[int, str]] = []        # 可追溯：(行号, 文本)
    in_fence = False

    def current_heading() -> str:
        # 结构感知：把标题栈拼成 "A > B > C" 面包屑
        return " > ".join(title for _, title in heading_stack)

    def flush() -> None:
        nonlocal section
        if section:
            chunks.extend(_emit_section(section, current_heading(), source_file))
            section = []

    # 可追溯：enumerate 给每行编号，section 里存 (行号, 文本)
    for line_no, raw in enumerate(text.splitlines(), start=1):
        # 结构感知：代码围栏内的 # 不是标题，需要跳过
        if raw.lstrip().startswith("```"):
            in_fence = not in_fence
            section.append((line_no, raw))
            continue

        # 结构感知：命中标题 → 收尾上一节、开新节
        if not in_fence:
            match = HEADING_RE.match(raw)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip().rstrip("#").strip("*`_").strip()
                flush()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                section.append((line_no, raw))  # 标题本身也作为块的上下文
                continue

        section.append((line_no, raw))

    flush()
    return chunks


def chunk_markdown_file(file_path: str) -> list[Chunk]:
    """读取项目内的 Markdown 文件并切块（复用 file_reader 的安全检查）。"""
    content = read_file_content(ReadFileInput(file_path=file_path))
    return chunk_text(content, file_path)


def chunk_directory(directory: str) -> list[Chunk]:
    """递归切分目录下所有受支持的文本文件，返回拼接后的块列表。"""
    base = (PROJECT_ROOT / directory).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"目录不存在：{directory}")

    chunks: list[Chunk] = []
    for root, dirs, files in os.walk(base):
        # 剪枝：把被屏蔽的目录从本轮遍历中移除，既不进入、也不列出其内文件
        dirs[:] = [d for d in dirs if d not in BLOCKED_DIRECTORY_NAMES]
        for name in files:
            file_path = Path(root) / name
            if file_path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            relative = file_path.relative_to(PROJECT_ROOT).as_posix()
            chunks.extend(chunk_markdown_file(relative))
    return chunks

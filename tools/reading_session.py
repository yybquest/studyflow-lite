from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from tools.config import PROJECT_ROOT
from tools.section_parser import Section

# 保留最近多少轮历史（user + assistant 各一条算一轮）
HISTORY_WINDOW = 10


@dataclass
class ReadingState:
    """一篇论文的精读进度（持久化到 knowledge/reading-state/<slug>.json）。"""

    paper: str
    current_section: int
    explained: list[int] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    updated_at: str = ""
    fallback: bool = False


def _slug(paper: str) -> str:
    """把论文相对路径变成安全的文件名：去扩展名，把路径分隔符换成短横线。"""
    text = paper.replace("\\", "/")
    text = re.sub(r"\.[^./]+$", "", text)
    return text.replace("/", "-")


def state_path(paper: str) -> Path:
    return PROJECT_ROOT / "knowledge" / "reading-state" / f"{_slug(paper)}.json"


def load_state(paper: str) -> ReadingState | None:
    path = state_path(paper)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ReadingState(
        paper=data.get("paper", paper),
        current_section=data.get("current_section", 0),
        explained=data.get("explained", []),
        history=data.get("history", []),
        updated_at=data.get("updated_at", ""),
        fallback=data.get("fallback", False),
    )


def save_state(state: ReadingState) -> Path:
    path = state_path(state.paper)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 只保留最近若干轮，避免 state 文件随对话无限增长（与 build_messages 的窗口一致）
    state.history = state.history[-(HISTORY_WINDOW * 2):]
    state.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.write_text(
        json.dumps(asdict(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


# 「复习」允许紧跟概念（中文里常不写空格），这些分隔符会被剥掉
_REVIEW_SEPARATORS = "：:，,、 "


def parse_command(raw: str) -> tuple[str, str]:
    """把用户输入解析成 (action, payload)。

    命中固定命令返回对应动作；否则视为针对当前节的提问（question）。
    - 「复习」允许紧跟概念（如「复习工具库这个概念」），也接受「复习：X」「复习一下 X」；
    - 「学 / 开始学」必须带空格，避免把「学习…」误判成命令。
    """
    text = raw.strip()
    if not text:
        return "question", ""
    if text.casefold() in {"exit", "退出", "quit"}:
        return "exit", ""

    if text == "复习":
        return "review", ""
    if text.startswith("复习"):
        payload = text[len("复习"):].strip().lstrip(_REVIEW_SEPARATORS).strip()
        if payload.startswith("一下"):
            payload = payload[len("一下"):].strip()
        return "review", payload

    # 「学 / 开始学」必须带空格，避免把「学习…」误判成命令
    for prefix, action in (("开始学", "study"), ("学", "study")):
        if text == prefix:
            return action, ""
        if text.startswith(prefix + " ") or text.startswith(prefix + "　"):
            return action, text[len(prefix):].strip()

    if text == "下一节":
        return "next", ""
    if text == "存笔记":
        return "save", ""

    return "question", text


def _section_line(section: Section) -> str:
    return f"{section.index}. {section.title}（第 {section.start_line}-{section.end_line} 行）"


def build_system_prompt(
    paper_text: str,
    sections: list[Section],
    current_index: int,
    fallback: bool = False,
) -> str:
    section_list = "\n".join(_section_line(section) for section in sections)
    current = sections[current_index] if 0 <= current_index < len(sections) else None
    current_desc = _section_line(current) if current else "无"
    fallback_note = (
        "\n注意：这篇论文没有可靠的标题分节，当前节按字数兜底切分。\n"
        if fallback else ""
    )
    return (
        "你是一个陪用户精读论文的学习助手。\n"
        "论文全文如下（Markdown）：\n"
        "-----\n"
        f"{paper_text}\n"
        "-----\n"
        f"节列表：\n{section_list}\n\n"
        f"当前节：{current_desc}\n"
        f"{fallback_note}\n"
        "规则：\n"
        "1. 用户要求讲解某节时，先用一两句话说明该节在全文中的作用，再逐段讲解原文，保留关键英文术语并解释；\n"
        "2. 用户提问时，只依据论文内容回答，并引用具体节与行号；论文里没有就说「论文里没有提到」；\n"
        "3. 讲完一节后停下，不要自动进入下一节。\n"
    )


def build_messages(
    state: ReadingState,
    sections: list[Section],
    paper_text: str,
    turn: str,
) -> list[dict]:
    messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                paper_text, sections, state.current_section, state.fallback
            ),
        }
    ]
    messages.extend(state.history[-(HISTORY_WINDOW * 2):])
    messages.append({"role": "user", "content": turn})
    return messages


def explain_turn(section: Section) -> str:
    return f"请讲解当前节【{section.title}】（第 {section.start_line}-{section.end_line} 行）。"


def _hit_location(hit: dict) -> str:
    """把检索块的位置拼成可读的来源描述。"""
    source = hit.get("source_file") or "未知来源"
    heading = hit.get("heading") or ""
    start = hit.get("start_line")
    end = hit.get("end_line")
    location = source
    if heading:
        location += f" / {heading}"
    if start is not None and end is not None:
        location += f"（第 {start}-{end} 行）"
    return location


def build_review_prompt(concept: str, hits: list[dict]) -> str:
    """把检索到的块组装成「复习某个概念」的提示词。"""
    blocks = [
        f"[{index}] 来源：{_hit_location(hit)}\n{hit.get('text', '').strip()}"
        for index, hit in enumerate(hits, start=1)
    ]
    return (
        f"请复习/解释概念【{concept}】。\n"
        "下面是从本地知识库检索到的相关片段。只依据这些片段作答，"
        "标注引用的来源编号；若片段不足以解释，就明确说「知识库里的片段不足以解释这个概念」。\n\n"
        + "\n\n".join(blocks)
    )

"""交互式精读入口：逐节讲解、自由提问、exit 后能续读。

用法：
    python reader.py "article/Role-Specialized Tool Agents.md"
"""
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from tools.file_reader import ReadFileInput, read_file_content
from tools.reading_session import (
    ReadingState,
    build_messages,
    explain_turn,
    load_state,
    parse_command,
    save_state,
)
from tools.section_parser import Section, fallback_sections, parse_sections


def _build_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("请在项目根目录的 .env 中设置 DEEPSEEK_API_KEY。")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def _chat(client: OpenAI, messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat"),
        messages=messages,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return response.choices[0].message.content or ""


def _record(state: ReadingState, turn: str, reply: str) -> None:
    state.history.append({"role": "user", "content": turn})
    state.history.append({"role": "assistant", "content": reply})
    save_state(state)


def _explain(client, state, sections, paper_text) -> None:
    section = sections[state.current_section]
    turn = explain_turn(section)
    try:
        reply = _chat(client, build_messages(state, sections, paper_text, turn))
    except Exception as error:  # 单次调用失败不应中断整个会话
        print(f"\n⚠️ 调用模型失败：{type(error).__name__}: {error}")
        print("（若是上下文超长，可换更短的论文，或等后续加「分节降级」模式。）\n")
        return
    print(f"\n【{section.title}】\n{reply}\n")
    if section.index not in state.explained:
        state.explained.append(section.index)
    _record(state, turn, reply)


def _answer(client, state, sections, paper_text, question: str) -> None:
    try:
        reply = _chat(client, build_messages(state, sections, paper_text, question))
    except Exception as error:  # 单次调用失败不应中断整个会话
        print(f"\n⚠️ 调用模型失败：{type(error).__name__}: {error}\n")
        return
    print(f"\n{reply}\n")
    _record(state, question, reply)


def _load_sections(paper: str) -> tuple[str, list[Section], bool]:
    paper_text = read_file_content(ReadFileInput(file_path=paper))
    sections = parse_sections(paper_text)
    if sections:
        return paper_text, sections, False
    return paper_text, fallback_sections(paper_text), True


def run(paper: str) -> None:
    client = _build_client()
    try:
        paper_text, sections, fallback = _load_sections(paper)
    except (ValueError, FileNotFoundError) as error:
        print(f"读取论文失败：{error}")
        return
    if not sections:
        print("无法从这篇文档解析出任何内容。")
        return
    if fallback:
        print("⚠️ 这篇没有可靠分节，将按字数切。")

    state = load_state(paper)
    if state is None:
        state = ReadingState(paper=paper, current_section=0, fallback=fallback)
        save_state(state)
        print(f"开始精读：{paper}")
        print(f"共 {len(sections)} 节：")
        for section in sections:
            print(f"  {section.index}. {section.title}")
        _explain(client, state, sections, paper_text)
    else:
        index = min(state.current_section, len(sections) - 1)
        print(f"续读：上次读到第 {index} 节【{sections[index].title}】。")
        print("输入「下一节」继续，或直接提问。")

    print("命令：学 <论文> / 下一节 / 存笔记 / 复习 <概念> / exit")
    try:
        while True:
            raw = input("你：").strip()
            if not raw:
                continue
            action, payload = parse_command(raw)
            if action == "exit":
                save_state(state)
                print("已保存进度，下次可续读。")
                break
            if action == "next":
                if state.current_section + 1 >= len(sections):
                    print("已到最后一节。可以说「存笔记」或 exit。")
                    continue
                state.current_section += 1
                save_state(state)
                _explain(client, state, sections, paper_text)
            elif action == "study":
                print("已经在精读中。直接说「下一节」或提问。")
            elif action == "save":
                print("（存笔记将在 M5 接入）")
            elif action == "review":
                if not payload:
                    print("要复习什么概念？")
                else:
                    print("（复习将在 M6 接入）")
            else:
                _answer(client, state, sections, paper_text, payload)
    except (EOFError, KeyboardInterrupt):
        save_state(state)
        print("\n已保存进度，退出。")


if __name__ == "__main__":
    paper = " ".join(sys.argv[1:]).strip()
    if not paper:
        print('用法：python reader.py "article/某论文.md"')
        sys.exit(1)
    run(paper)

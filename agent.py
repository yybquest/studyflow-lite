import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from tools.file_reader import ReadFileInput, read_file_tool
from tools.file_searcher import SearchFilesInput, search_files_tool
from tools.search_text import SearchTextInput, search_text_tool

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise SystemExit("请在项目根目录的 .env 文件中设置 DEEPSEEK_API_KEY。")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_local_documents",
            "description": (
                "仅当用户明确要查找某个关键词、短语、概念或定位相关段落时使用。"
                "它只返回匹配行及周边上下文，不会返回整篇文件。若用户要读取、总结、"
                "翻译或概述指定文档的整体内容，应改用 read_document。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_path": {
                        "type": "string",
                        "description": "项目根目录下的相对路径。通常使用 notes；也可以指定单个文件，例如 notes/day-1.md。",
                    },
                    "search_key_word": {
                        "type": "string",
                        "description": "用户明确要求查找的关键词或短语，也可以从问题中提取关键词；但不要把整句问题当作关键词。",
                    },
                    "file_suffix": {
                        "type": "string",
                        "description": "可选文件后缀，例如 .md 或 .txt；不确定时使用 .md。",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最多返回多少条匹配结果，建议 5。",
                        "minimum": 1,
                        "maximum": 10,
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "每条命中前后附带的上下文行数，建议 2。",
                        "minimum": 0,
                        "maximum": 5,
                    },
                },
                "required": ["search_path", "search_key_word"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "询问目录里有哪些文件时，调用 list_documents。",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_path": {
                        "type": "string",
                        "description": "项目根目录下的相对目录，例如 notes",
                    },
                    "file_suffix": {
                        "type": "string",
                        "description": "可选文件后缀，例如 .md；省略时列出所有支持的文档文件。",
                    },
                },
                "required": ["search_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": (
                "读取项目内指定文本文件的完整内容，或按行号范围读取部分内容。"
                "用户要求阅读、展示、总结、概述、翻译或分析某个文档整体内容时使用；"
                "即使用户没有提供关键词，也应使用此工具。不要用它代替关键词搜索。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "项目根目录下的相对文件路径，例如 notes/day-2.md。",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "可选的起始行号，从 1 开始；省略时从文件开头读取。",
                        "minimum": 1,
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "可选的结束行号，包含该行；省略时读取到文件末尾。",
                        "minimum": 1,
                    },
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
    },
]


SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "你是一个本地笔记助手。用户询问目录中有哪些文件时，调用 list_documents。"
        "用户明确要求查找关键词、短语、概念或定位相关段落时，调用 search_local_documents；"
        "用户要求读取、展示、总结、概述、翻译或分析某个文档整体内容时，先调用 read_document，"
        "再依据读取结果完成任务。不要用关键词搜索结果代替全文阅读。"
        "回答文档内容问题时只依据工具结果；找不到依据就明确说明，不要编造，并引用来源文件和行号。"
    ),
}


def run_agent(question: str, messages: list[dict]) -> None:
    """在当前会话中回答一个问题，并保留后续对话所需的消息历史。"""
    messages.append({"role": "user", "content": question})

    # 限制往返次数，避免模型反复调用工具而不结束。
    for _ in range(5):
        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            extra_body={"thinking": {"type": "disabled"}},
        )
        assistant_message = response.choices[0].message

        # 把 assistant 的工具调用消息原样放回对话，满足 API 工具调用协议。
        messages.append(assistant_message.model_dump(exclude_none=True))

        if not assistant_message.tool_calls:
            print(f"智能体：{assistant_message.content or '模型没有返回文本。'}")
            return

        # Python 程序执行工具；模型只负责提出调用和参数，不会直接访问本地文件。
        
        for tool_call in assistant_message.tool_calls:

            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if tool_call.function.name == "search_local_documents":
                request = SearchTextInput.model_validate(arguments)
                matches = search_text_tool(request)
                result = [match.model_dump() for match in matches]

            elif tool_call.function.name == "list_documents":
                request = SearchFilesInput.model_validate(arguments)
                result = search_files_tool(request)

            elif tool_call.function.name == "read_document":
                request = ReadFileInput.model_validate(arguments)
                result = read_file_tool(request)
                
            else:
                result = {"error": f"未知工具：{tool_name}"}

            tool_result = json.dumps(result, ensure_ascii=False)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    print("工具调用轮数已达上限，请缩小问题范围后重试。")


if __name__ == "__main__":
    messages = [SYSTEM_MESSAGE]
    first_question = " ".join(sys.argv[1:]).strip()

    if first_question:
        run_agent(first_question, messages)

    print("本地笔记助手已启动。输入 exit 结束。")
    try:
        while True:
            question = input("你：").strip()
            if question.casefold() == "exit":
                print("对话结束。")
                break
            if question:
                run_agent(question, messages)
    except (EOFError, KeyboardInterrupt):
        print("\n对话结束。")
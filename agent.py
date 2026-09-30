import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from tools.search_text_tool import SearchTextInput, search_text_tool

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
            "description": "在项目内指定的目录或单个文本文件中搜索关键词，返回匹配行、上下文、文件路径和行号。回答本地笔记问题时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_path": {
                        "type": "string",
                        "description": "项目根目录下的相对路径。通常使用 notes；也可以指定单个文件，例如 notes/day-1.md。",
                    },
                    "search_key_word": {
                        "type": "string",
                        "description": "从用户问题提取的关键词或短语。",
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
    }
]


def run_agent(question: str) -> None:
    """让模型决定何时搜索本地笔记，并基于搜索结果作答。"""
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个本地笔记问答助手。回答关于本地笔记的问题前，必须先调用 "
                "search_local_documents 搜索。只根据工具返回的内容回答；如果没有找到依据，"
                "明确说笔记中没有找到答案，不要编造。回答时引用来源文件和行号。"
            ),
        },
        {"role": "user", "content": question},
    ]

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
            print(assistant_message.content or "模型没有返回文本。")
            return

        # Python 程序执行工具；模型只负责提出调用和参数，不会直接访问本地文件。
        for tool_call in assistant_message.tool_calls:
            if tool_call.function.name != "search_local_documents":
                tool_result = json.dumps(
                    {"error": f"未知工具：{tool_call.function.name}"},
                    ensure_ascii=False,
                )
            else:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    request = SearchTextInput.model_validate(arguments)
                    matches = search_text_tool(request)
                    tool_result = json.dumps(
                        [match.model_dump() for match in matches],
                        ensure_ascii=False,
                    )
                except (json.JSONDecodeError, ValidationError, OSError, ValueError) as error:
                    tool_result = json.dumps(
                        {"error": str(error)},
                        ensure_ascii=False,
                    )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    print("工具调用轮数已达上限，请缩小问题范围后重试。")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("请输入关于本地笔记的问题：").strip()
    if question:
        run_agent(question)
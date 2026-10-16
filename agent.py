import json
import os
import sys

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from pydantic import ValidationError

from tools.file_reader import ReadFileInput, read_file_tool
from tools.file_searcher import SearchFilesInput, search_files_tool
from tools.search_text import SearchTextInput, search_text_tool
from tools.create_markdown import CreateMarkdownInput, create_markdown_tool
from tools.edit_markdown import EditMarkdownInput, edit_markdown_tool
from tools.pdf_to_text import PdfToTextInput, pdf_to_text_tool
from tools.search_knowledge import SearchKnowledgeInput, search_knowledge_tool
from tool_schema import to_function_schema

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
ds_model_name = os.getenv("DEEPSEEK_MODEL_NAME")
if not api_key:
    raise SystemExit("请在项目根目录的 .env 文件中设置 DEEPSEEK_API_KEY。")

if not ds_model_name:
    raise SystemExit("请在项目根目录的 .env 文件中设置 DEEPSEEK_MODEL_NAME。")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

# 名称 -> (工具说明「何时用」, 输入模型, 执行函数)
TOOL_SPECS = {
    "search_keyword": (
        "仅当用户明确要查找某个关键词、短语、概念或定位相关段落时使用。"
        "它只返回匹配行及周边上下文，不会返回整篇文件。若用户要读取、总结、"
        "翻译或概述指定文档的整体内容，应改用 read_document。",
        SearchTextInput,
        search_text_tool,
    ),
    "search_knowledge": (
        "在本地知识库中做语义检索，返回最相关的文档片段及来源。"
        "用于回答需要结合笔记或论文内容来理解的开放问题。"
        "它只返回片段、不生成答案；复杂问题要拆成子问题、用不同关键词多次调用。",
        SearchKnowledgeInput,
        search_knowledge_tool,
    ),
    "list_documents": (
        "询问目录里有哪些文件时，调用 list_documents。"
        "只给出文件名时应先用该工具找到文件路径，确认文件名，再进行下一步操作。"
        "列出的 .pdf 文件需要用 read_pdf 处理，其他文本文件用 read_document 读取。",
        SearchFilesInput,
        search_files_tool,
    ),
    "read_document": (
        "读取项目内指定文本文件的完整内容，或按行号范围读取部分内容。"
        "返回内容每行形如 `行号 | 内容`，行号从 1 开始，空行也会带行号。"
        "用户要求阅读、展示、总结、概述、翻译或分析某个文档整体内容时使用；"
        "即使用户没有提供关键词，也应使用此工具。不要用它代替关键词搜索。",
        ReadFileInput,
        read_file_tool,
    ),
    "create_markdown": (
        "在项目内创建 UTF-8 Markdown 文件；若文件已存在则拒绝覆盖。",
        CreateMarkdownInput,
        create_markdown_tool,
    ),
    "edit_markdown": (
        "直接修改项目内已存在的 Markdown 文件，支持插入、替换和删除。"
        "调用成功后文件已经写入；该工具不会创建文件。",
        EditMarkdownInput,
        edit_markdown_tool,
    ),
    "pdf_to_text":(
        "把项目内的 PDF 论文抽取为结构化的纯文本，并保存为 .md 文件。"
        "之后可以用 read_document 读取、或 search_keyword 检索该 .md 文件，"
        "不要直接读取 PDF 原文。",
        PdfToTextInput,
        pdf_to_text_tool,
    ),
}

TOOLS = [
    to_function_schema(name, description, model)
    for name, (description, model, _func) in TOOL_SPECS.items()
]

# 工具注册表：名称 -> (输入模型, 执行函数)
TOOL_IMPLEMENTATIONS = {
    name: (model, func)
    for name, (_description, model, func) in TOOL_SPECS.items()
}

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "你是一个本地笔记助手。用户询问目录中有哪些文件时，调用 list_documents。"
        "用户明确要求查找关键词、短语、概念或定位相关段落时，调用 search_keyword。"
        "用户提出需要结合笔记或论文内容来回答的开放问题时，调用 search_knowledge 做语义检索；"
        "复杂问题要拆成多个子问题、用不同关键词多次调用 search_knowledge，检索到相关块后如有必要再用 read_document 读完整章节。"
        "用户要求读取、展示、总结、概述、翻译或分析某个文档整体内容时，先调用 read_document。"
        "用户要求阅读 PDF 论文时，先调用 pdf_to_text 把 PDF 抽成文本，再进行下一步处理。"
        "再依据读取结果完成任务。不要用关键词搜索结果代替全文阅读。"
        "用户要求创建 Markdown 文件时，调用 create_markdown。"
        "用户明确要求编辑 Markdown 文件时，调用 edit_markdown；工具成功返回后，文件已直接修改。"
        "生成 Markdown 内容时保持规范结构：标题或加粗短语单独成行，段落之间换行分隔。"
        "回答文档内容问题时只依据工具结果；找不到依据就明确说明，不要编造，并引用来源文件和行号。"
    ),
}


def _execute_tool_call(tool_call) -> str:
    """执行单个工具调用；任何错误都转成 JSON 回传给模型，而不是让进程崩溃。"""
    name = tool_call.function.name
    implementation = TOOL_IMPLEMENTATIONS.get(name)
    if implementation is None:
        return json.dumps({"error": f"未知工具：{name}"}, ensure_ascii=False)

    input_model, tool_func = implementation
    try:
        arguments = json.loads(tool_call.function.arguments)
        request = input_model.model_validate(arguments)
        result = tool_func(request)
    except json.JSONDecodeError as error:
        # 模型给出的参数不是合法 JSON
        return json.dumps({"error": f"参数不是合法 JSON：{error}"}, ensure_ascii=False)
    except ValidationError as error:
        # pydantic 校验失败（缺字段、类型错、越界等）
        return json.dumps({"error": f"参数校验失败：{error}"}, ensure_ascii=False)
    except (FileNotFoundError, ValueError) as error:
        # 工具内部的业务错误：文件不存在、路径越界、文件已存在等
        return json.dumps({"error": str(error)}, ensure_ascii=False)
    except Exception as error:
        # 兜底，保证 agent 永不因单个工具而崩溃
        return json.dumps({"error": f"工具执行失败：{type(error).__name__}: {error}"}, ensure_ascii=False)

    # 统一把 pydantic 模型（如 SearchTextMatch）转成 dict，方便 JSON 序列化
    if isinstance(result, list) and result and hasattr(result[0], "model_dump"):
        result = [item.model_dump() for item in result]
    return json.dumps(result, ensure_ascii=False)

def run_agent(question: str, messages: list[dict]) -> None:
    """在当前会话中回答一个问题，并保留后续对话所需的消息历史。"""
    messages.append({"role": "user", "content": question})

    # 限制往返次数，避免模型反复调用工具而不结束。
    for _ in range(8):
        try:
            response = client.chat.completions.create(
                model=ds_model_name,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                extra_body={"thinking": {"type": "disabled"}},
            )
        except APITimeoutError:
            print("智能体：请求超时，请稍后重试。")
            return
        except RateLimitError:
            print("智能体：请求太频繁，被限流了，请稍后重试。")
            return
        except APIConnectionError:
            print("智能体：网络连接失败，请检查网络后重试。")
            return
        except AuthenticationError:
            print("智能体：API Key 无效，请检查 .env 里的 DEEPSEEK_API_KEY。")
            return
        except OpenAIError as error:
            # 兜底：其余所有 openai 异常
            print(f"智能体：请求失败：{error}")
            return

        # 把 assistant 的工具调用消息原样放回对话，满足 API 工具调用协议。
        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_none=True))

        if not assistant_message.tool_calls:
            print(f"智能体：{assistant_message.content or '模型没有返回文本。'}")
            return

        # Python 程序执行工具；模型只负责提出调用和参数，不会直接访问本地文件。
        for tool_call in assistant_message.tool_calls:
            tool_result = _execute_tool_call(tool_call)
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
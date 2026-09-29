from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from tools.file_reader import ReadFileInput, read_file_tool
from tools.file_searcher import (
    SUPPORTED_SUFFIXES,
    SearchFilesInput,
    search_files_tool,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class SearchTextInput(BaseModel):
    """搜索文本工具的结构化输入。"""

    search_path: str
    search_key_word: str
    file_suffix: str | None = Field(default=None)
    max_results: int = Field(default=50, ge=1, le=500)
    context_lines: int = Field(default=2, ge=0, le=10)

    @model_validator(mode="after")
    def validate_search_path(self) -> "SearchTextInput":
        """确保搜索路径不是空字符串，并且文件后缀是支持的类型。"""
        if not self.search_path.strip():
            raise ValueError("search_path 不能为空")
        if not self.search_key_word.strip():
            raise ValueError("search_key_word 不能为空")
        if (
            self.file_suffix is not None
            and self.file_suffix.lower() not in SUPPORTED_SUFFIXES
        ):
            raise ValueError(f"不支持的文件后缀: {self.file_suffix}")
        return self


class SearchTextMatch(BaseModel):
    """单条文本搜索结果，包含来源、命中行和周边上下文。"""

    file_path: str
    line_number: int
    text: str
    context_start_line: int
    context: str


def search_text_tool(request: SearchTextInput) -> list[SearchTextMatch]:
    """在目录或单个文件中搜索关键词，并返回命中行与上下文。"""
    target_path = (PROJECT_ROOT / request.search_path).resolve()

    # 1) 如果传入的是文件，则只搜索这个文件；否则按目录搜索。
    if request.search_path and target_path.is_file():
        file_paths = [request.search_path]
    else:
        # 复用文件枚举工具；没有指定后缀时，逐个枚举受支持的文本类型。
        suffixes = (
            [request.file_suffix.lower()]
            if request.file_suffix is not None
            else sorted(SUPPORTED_SUFFIXES)
        )
        file_paths = []
        for suffix in suffixes:
            file_paths.extend(
                search_files_tool(
                    SearchFilesInput(
                        search_path=request.search_path,
                        file_suffix=suffix,
                    )
                )
            )

    # 复用文件读取工具；路径安全、文件类型和大小检查都由它负责。
    matches: list[SearchTextMatch] = []
    keyword = request.search_key_word.casefold()
    for file_path in sorted(set(file_paths)):
        content = read_file_tool(ReadFileInput(file_path=file_path))

        lines = content.splitlines()
        for line_index, line in enumerate(lines):
            if keyword in line.casefold():
                context_start_index = max(0, line_index - request.context_lines)
                context_end_index = min(
                    len(lines), line_index + request.context_lines + 1
                )
                matches.append(
                    SearchTextMatch(
                        file_path=file_path,
                        line_number=line_index + 1,
                        text=line,
                        context_start_line=context_start_index + 1,
                        context="\n".join(lines[context_start_index:context_end_index]),
                    )
                )
                if len(matches) >= request.max_results:
                    return matches

    return matches


def format_search_matches(matches: list[SearchTextMatch]) -> str:
    """将搜索结果格式化为便于终端阅读的多行文本。"""
    if not matches:
        return "没有找到匹配内容。"

    # 每条结果分块显示，并用 > 标记命中行，其余行为上下文。
    return "\n\n".join(
        f"[{index}] {match.file_path}（命中第 {match.line_number} 行）\n"
        + "\n".join(
            f"{'>' if match.context_start_line + offset == match.line_number else ' '} "
            f"{match.context_start_line + offset}: {line}"
            for offset, line in enumerate(match.context.splitlines())
        )
        for index, match in enumerate(matches, start=1)
    )


from pathlib import Path

from pydantic import BaseModel, Field, model_validator

# 工具默认只能读取项目根目录中的文件，避免意外访问项目外的路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 只允许搜索常见文本格式；PDF、Word 等格式需要专门的解析库。
SUPPORTED_SUFFIXES = {".txt", ".md", ".py", ".json", ".csv", ".yaml", ".yml"}

# 这些目录通常包含依赖、版本控制数据或缓存，不应作为文档来源。
BLOCKED_DIRECTORY_NAMES = {".git", ".venv", "venv", "__pycache__"}

class SearchFilesInput(BaseModel):
    """搜索文件工具的结构化输入。"""

    search_path: str = Field(
        description="项目根目录下的相对文件或目录路径，不允许绝对路径，例如 notes。"
    )
    file_suffix: str | None = Field(
        default=None,
        description="可选的文件后缀，例如 .md；省略时搜索所有支持的文本文件。",
    )

    @model_validator(mode="after")
    def validate_search_path(self) -> "SearchFilesInput":
        """确保搜索路径不是空字符串，并且文件后缀是支持的类型。"""
        if not self.search_path.strip():
            raise ValueError("search_path 不能为空")
        if self.file_suffix is not None and self.file_suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"不支持的文件后缀: {self.file_suffix}")
        return self

def search_files_tool(request: SearchFilesInput) -> list[str]:
    """搜索项目内的文件，返回匹配的相对路径列表。"""
    requested_path = Path(request.search_path)

    # 不接受绝对路径，工具调用方只能指定项目内的相对路径。
    if requested_path.is_absolute():
        raise ValueError("请提供项目根目录下的相对路径")

    # resolve 会处理 .. 和符号链接；relative_to 用来确认最终路径仍在项目内。
    resolved_path = (PROJECT_ROOT / requested_path).resolve()
    try:
        resolved_path.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError("不允许搜索项目目录以外的文件") from error

    # 阻止搜索依赖、缓存和版本控制目录中的文件。
    if BLOCKED_DIRECTORY_NAMES.intersection(resolved_path.parts):
        raise ValueError("不允许搜索依赖、缓存或版本控制目录中的文件")

    # 搜索文件并过滤后缀
    matched_files = []
    for file in resolved_path.rglob("*"):
        if file.is_file() and (
            file.suffix.lower() in SUPPORTED_SUFFIXES
            if request.file_suffix is None
            else file.suffix.lower() == request.file_suffix.lower()):
            matched_files.append(file.relative_to(PROJECT_ROOT).as_posix())

    return matched_files
from pathlib import Path
from difflib import unified_diff
from typing import Literal

from pydantic import BaseModel, Field, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_SIZE_BYTES = 1_000_000
BLOCKED_DIRECTORY_NAMES = {".git", ".venv", "venv", "__pycache__"}


class EditMarkdownInput(BaseModel):
    """修改 Markdown 文件的结构化输入。"""

    file_path: str = Field(
        description="项目根目录下的相对文件路径，必须以 .md 结尾，例如 notes/day-1.md。"
    )
    action: Literal["insert", "replace", "delete"] = Field(
        description="修改类型：insert 插入内容、replace 替换文本、delete 删除文本。"
    )
    content: str | None = Field(
        default=None,
        description="插入或替换时使用的新内容；删除操作时可忽略。",
    )
    target_text: str | None = Field(
        default=None,
        description="替换或删除前，目标原文片段。对于替换必须提供；删除时可提供唯一原文片段或行范围。",
    )
    line_number: int | None = Field(
        default=None,
        ge=1,
        description="插入时指定的目标行号；省略时在文件末尾追加。",
    )
    start_line: int | None = Field(
        default=None,
        ge=1,
        description="按范围修改时的起始行号，从 1 开始。",
    )
    end_line: int | None = Field(
        default=None,
        ge=1,
        description="按范围修改时的结束行号，包含该行。",
    )
    preview_only: bool = Field(
        default=True,
        description="是否仅展示变更预览而不写入文件；默认 True，通常先预览后再确认。",
    )

    @model_validator(mode="after")
    def validate_action(self) -> "EditMarkdownInput":
        """确保动作和参数满足编辑语义，并保证目标路径是安全的 Markdown 文件。"""
        if not self.file_path.strip():
            raise ValueError("file_path 不能为空")

        requested_path = Path(self.file_path)
        if requested_path.is_absolute():
            raise ValueError("请提供项目根目录下的相对路径")
        if requested_path.suffix.lower() != ".md":
            raise ValueError("只允许修改 .md 文件")

        if self.action == "insert":
            if self.content is None:
                raise ValueError("insert 操作必须提供 content")
        elif self.action == "replace":
            if self.content is None:
                raise ValueError("replace 操作必须提供 content")
            if self.target_text is None and (self.start_line is None or self.end_line is None):
                raise ValueError("replace 操作需要提供 target_text 或起止行号范围")
        elif self.action == "delete":
            if self.target_text is None and (self.start_line is None or self.end_line is None):
                raise ValueError("delete 操作需要提供 target_text 或起止行号范围")

        if self.start_line is not None and self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line 不能小于 start_line")

        return self


def _resolve_project_file(file_path: str) -> Path:
    """把相对路径解析为项目内安全文件路径。"""
    requested_path = Path(file_path)
    resolved_path = (PROJECT_ROOT / requested_path).resolve()
    try:
        resolved_path.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError("不允许修改项目目录以外的文件") from error

    if BLOCKED_DIRECTORY_NAMES.intersection(resolved_path.parts):
        raise ValueError("不允许修改依赖、缓存或版本控制目录中的文件")

    return resolved_path


def _build_diff(old_text: str, new_text: str) -> str:
    """生成精简的统一 diff，用于展示修改前后差异。"""
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    diff = list(
        unified_diff(
            old_lines,
            new_lines,
            fromfile="original",
            tofile="modified",
            lineterm="",
        )
    )
    return "\n".join(diff)


def edit_markdown_tool(request: EditMarkdownInput) -> dict[str, str | int | bool | list[str]]:
    """安全修改项目内的 Markdown 文件，默认先返回预览 diff 再决定是否落盘。"""
    resolved_path = _resolve_project_file(request.file_path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"文件不存在：{request.file_path}")

    if not resolved_path.is_file():
        raise ValueError(f"目标不是文件：{request.file_path}")

    content = resolved_path.read_text(encoding="utf-8")
    original_content = content
    lines = content.splitlines(keepends=True)

    if request.action == "insert":
        insert_text = request.content or ""
        if request.line_number is None:
            new_content = content + insert_text
        else:
            target_index = max(0, request.line_number - 1)
            if target_index > len(lines):
                raise ValueError("line_number 超出文件范围")
            lines.insert(target_index, insert_text)
            new_content = "".join(lines)

    elif request.action == "replace":
        if request.start_line is not None or request.end_line is not None:
            start_index = (request.start_line or 1) - 1
            end_index = request.end_line if request.end_line is not None else len(lines)
            if start_index < 0 or end_index > len(lines):
                raise ValueError("起止行号超出文件范围")
            replacement = request.content or ""
            head = "".join(lines[:start_index])
            tail = "".join(lines[end_index:])
            new_content = head + replacement + tail
        else:
            target = request.target_text
            if target is None:
                raise ValueError("replace 操作必须提供 target_text 或行范围")
            if target not in content:
                raise ValueError(f"未找到需要替换的文本：{target[:50]}")
            new_content = content.replace(target, request.content or "", 1)

    elif request.action == "delete":
        if request.start_line is not None or request.end_line is not None:
            start_index = (request.start_line or 1) - 1
            end_index = request.end_line if request.end_line is not None else len(lines)
            if start_index < 0 or end_index > len(lines):
                raise ValueError("起止行号超出文件范围")
            new_content = "".join(lines[:start_index] + lines[end_index:])
        else:
            target = request.target_text
            if target is None:
                raise ValueError("delete 操作必须提供 target_text 或行范围")
            if content.count(target) == 0:
                raise ValueError(f"未找到需要删除的文本：{target[:50]}")
            if content.count(target) > 1:
                raise ValueError("待删除文本出现多次，需改为指定行范围或更具体的唯一文本")
            new_content = content.replace(target, "", 1)

    else:
        raise ValueError(f"不支持的动作：{request.action}")

    if len(new_content.encode("utf-8")) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"修改后的内容超过 {MAX_FILE_SIZE_BYTES} 字节的写入上限：{request.file_path}"
        )

    diff_text = _build_diff(original_content, new_content)
    preview = {
        "status": "preview" if request.preview_only else "updated",
        "file_path": resolved_path.relative_to(PROJECT_ROOT).as_posix(),
        "action": request.action,
        "preview_only": request.preview_only,
        "diff": diff_text,
        "before": original_content,
        "after": new_content,
    }

    if request.preview_only:
        return preview

    resolved_path.write_text(new_content, encoding="utf-8", newline="")
    return {
        **preview,
        "status": "updated",
        "bytes_written": len(new_content.encode("utf-8")),
    }

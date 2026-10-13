from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from tools.config import BLOCKED_DIRECTORY_NAMES, PROJECT_ROOT

MAX_PDF_SIZE_BYTES = 20_000_000


class PdfToTextInput(BaseModel):
    """把 PDF 抽成文本文件的结构化输入。"""

    file_path: str = Field(
        description="项目根目录下的相对 PDF 路径，例如 article/attention.pdf。"
    )
    output_path: str | None = Field(
        default=None,
        description="可选：抽取结果保存到的 .md 路径；省略时保存为与 PDF 同目录、同名的 .md 文件。",
    )

    @model_validator(mode="after")
    def validate_file_path(self) -> "PdfToTextInput":
        if not self.file_path.strip():
            raise ValueError("file_path 不能为空")
        if self.output_path is not None and not self.output_path.strip():
            raise ValueError("output_path 不能为空字符串")
        return self


def _resolve_pdf(file_path: str) -> Path:
    """把相对路径解析为项目内安全的 PDF 文件路径。"""
    requested = Path(file_path)
    if requested.is_absolute():
        raise ValueError("请提供项目根目录下的相对路径")

    resolved = (PROJECT_ROOT / requested).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError("不允许读取项目目录以外的文件") from error

    if BLOCKED_DIRECTORY_NAMES.intersection(resolved.parts):
        raise ValueError("不允许读取依赖、缓存或版本控制目录中的文件")

    if resolved.suffix.lower() != ".pdf":
        raise ValueError(f"只支持 .pdf 文件，收到：{resolved.suffix or '无扩展名'}")

    if not resolved.is_file():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    if resolved.stat().st_size > MAX_PDF_SIZE_BYTES:
        raise ValueError(f"文件超过 {MAX_PDF_SIZE_BYTES} 字节的读取上限：{file_path}")

    return resolved


def _extract_pdf_text(resolved_path: Path) -> tuple[str, int]:
    """内部辅助：提取 PDF 全部文本，返回 (文本, 总页数)。"""
    import pymupdf  # PyMuPDF，延迟导入
    import pymupdf4llm

    document = pymupdf.open(str(resolved_path))
    try:
        total_pages = document.page_count
        text = pymupdf4llm.to_markdown(document)
        return text, total_pages
    finally:
        document.close()


def pdf_to_text_tool(request: PdfToTextInput) -> dict:
    """把 PDF 抽成带页码标记的文本，保存为 .md，返回保存路径与统计信息。"""
    resolved_path = _resolve_pdf(request.file_path)

    if request.output_path is None:
        output_path = resolved_path.with_suffix(".md")
    else:
        output_path = (PROJECT_ROOT / request.output_path).resolve()
        try:
            output_path.relative_to(PROJECT_ROOT)
        except ValueError as error:
            raise ValueError("不允许在项目目录以外保存文件") from error
        if output_path.suffix.lower() != ".md":
            raise ValueError("output_path 必须以 .md 结尾")
        if BLOCKED_DIRECTORY_NAMES.intersection(output_path.parts):
            raise ValueError("不允许在依赖、缓存或版本控制目录中保存文件")

    text, total_pages = _extract_pdf_text(resolved_path)

    # 抽取结果是「派生物」，重跑可以覆盖，所以这里允许直接写。
    output_path.write_text(text, encoding="utf-8")

    return {
        "status": "created",
        "file_path": output_path.relative_to(PROJECT_ROOT).as_posix(),
        "pages": total_pages,
        "chars": len(text),
        "preview": text[:500],
    }
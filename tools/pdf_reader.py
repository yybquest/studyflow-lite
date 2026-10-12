from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from tools.config import BLOCKED_DIRECTORY_NAMES, PROJECT_ROOT

# PDF 是二进制文件，允许比纯文本更大的体积上限（20MB）。
MAX_PDF_SIZE_BYTES = 20_000_000


class ReadPdfInput(BaseModel):
    """读取 PDF 论文的结构化输入。"""

    file_path: str = Field(
        description="项目根目录下的相对 PDF 路径，例如 article/attention.pdf。"
    )
    start_page: int | None = Field(
        default=None,
        ge=1,
        description="可选起始页，从 1 开始；省略时从第 1 页读取。",
    )
    end_page: int | None = Field(
        default=None,
        ge=1,
        description="可选结束页，包含该页；省略时读到最后一页。",
    )

    @model_validator(mode="after")
    def validate_page_range(self) -> "ReadPdfInput":
        if not self.file_path.strip():
            raise ValueError("file_path 不能为空")
        if (
            self.start_page is not None
            and self.end_page is not None
            and self.end_page < self.start_page
        ):
            raise ValueError("end_page 不能小于 start_page")
        return self


def read_pdf_tool(request: ReadPdfInput) -> str:
    """读取项目内 PDF，按页码范围提取文本，返回带页码标记的文本。"""
    requested_path = Path(request.file_path)

    if requested_path.is_absolute():
        raise ValueError("请提供项目根目录下的相对路径")

    resolved_path = (PROJECT_ROOT / requested_path).resolve()
    try:
        resolved_path.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError("不允许读取项目目录以外的文件") from error

    if BLOCKED_DIRECTORY_NAMES.intersection(resolved_path.parts):
        raise ValueError("不允许读取依赖、缓存或版本控制目录中的文件")

    if resolved_path.suffix.lower() != ".pdf":
        raise ValueError(f"只支持 .pdf 文件，收到：{resolved_path.suffix or '无扩展名'}")

    if not resolved_path.is_file():
        raise FileNotFoundError(f"文件不存在：{request.file_path}")

    if resolved_path.stat().st_size > MAX_PDF_SIZE_BYTES:
        raise ValueError(f"文件超过 {MAX_PDF_SIZE_BYTES} 字节的读取上限：{request.file_path}")

    import pymupdf

    document = pymupdf.open(str(resolved_path))
    try:
        total_pages = document.page_count
        start_page = request.start_page or 1
        end_page = request.end_page or total_pages

        if start_page > total_pages:
            raise ValueError(f"start_page 超出总页数 {total_pages}")
        end_page = min(end_page, total_pages)

        pages = []
        for page_number in range(start_page, end_page + 1):
            page = document[page_number - 1]
            text = page.get_text(sort=True)  # sort=True 改善双栏论文的阅读顺序
            pages.append(f"===== 第 {page_number} 页 =====\n{text}")
        return "\n".join(pages)
    finally:
        document.close()